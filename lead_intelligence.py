#!/usr/bin/env python3
"""Lead Intelligence System: deterministic qualification + batched LLM explanations.

The weighted score and final decision are computed locally. The LLM may explain
those results and draft outreach for qualified leads, but it cannot change scores
or decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError, InternalServerError, RateLimitError
except ImportError:  # Keep deterministic/offline mode usable without the SDK.
    OpenAI = None  # type: ignore[assignment]
    APIConnectionError = APITimeoutError = APIStatusError = InternalServerError = RateLimitError = Exception  # type: ignore[misc,assignment]

REQUIRED_COLUMNS = ["name", "company", "company_size", "industry", "source", "last_interaction_date"]
QUALIFICATION_REQUIRED_FIELDS = ["name", "source", "last_interaction_date", "company_size"]

INTENT_SCORES = {
    "inbound demo request": 10,
    "referral": 9,
    "content download": 7,
    "webinar attendee": 6,
    "sales call": 5,
    "linkedin outreach": 4,
}

WEIGHTS = {"buying_intent": 0.40, "recency": 0.30, "company_size": 0.20, "data_quality": 0.10}
DATA_QUALITY_PENALTIES = {"name": 5, "source": 4, "last_interaction_date": 3, "company_size": 3}


def is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def parse_company_size(value: Any) -> int | None:
    if is_missing(value):
        return None
    try:
        size = int(str(value).strip())
        return size if size >= 0 else None
    except (TypeError, ValueError):
        return None


def parse_date(value: Any) -> date | None:
    if is_missing(value):
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return None


def round_score(value: float) -> int:
    """Round halves up (rather than Python's banker's rounding), clamp to 1..10."""
    return max(1, min(10, int(value + 0.5)))


def score_recency(days_since: int | None) -> int:
    if days_since is None:
        return 1
    if days_since < 0:
        # Future-dated interactions are invalid, not evidence of recency.
        return 1
    if days_since <= 7:
        return 10
    if days_since <= 30:
        return 8
    if days_since <= 90:
        return 5
    return 2


def score_lead(row: dict[str, Any], reference_date: date, input_index: int) -> dict[str, Any]:
    """Calculate all deterministic scores and the initial decision for one lead."""
    lead = {key: ("" if row.get(key) is None else str(row.get(key)).strip()) for key in REQUIRED_COLUMNS}

    missing_for_quality: list[str] = []
    for field in DATA_QUALITY_PENALTIES:
        if field == "company_size":
            if parse_company_size(lead[field]) is None:
                missing_for_quality.append(field)
        elif field == "last_interaction_date":
            if parse_date(lead[field]) is None:
                missing_for_quality.append(field)
        elif field == "source":
            if is_missing(lead[field]) or lead[field].casefold() not in INTENT_SCORES:
                missing_for_quality.append(field)
        elif is_missing(lead[field]):
            missing_for_quality.append(field)

    data_quality_score = max(1, 10 - sum(DATA_QUALITY_PENALTIES[field] for field in missing_for_quality))

    source_key = lead["source"].casefold()
    intent_score = INTENT_SCORES.get(source_key, 1)
    source_is_missing_or_unknown = source_key not in INTENT_SCORES

    interaction_date = parse_date(lead["last_interaction_date"])
    days_since = (reference_date - interaction_date).days if interaction_date else None
    # Future dates are invalid; treat as unknown for scoring and review.
    if days_since is not None and days_since < 0:
        days_since = None
    recency_score = score_recency(days_since)

    size = parse_company_size(lead["company_size"])
    company_size_score = 5 if size is None else (10 if size <= 500 else 6)

    raw_score = (
        WEIGHTS["buying_intent"] * intent_score
        + WEIGHTS["recency"] * recency_score
        + WEIGHTS["company_size"] * company_size_score
        + WEIGHTS["data_quality"] * data_quality_score
    )
    final_score = round_score(raw_score)

    review_flags: list[str] = []
    if is_missing(lead["name"]):
        review_flags.append("Missing contact name")
    if source_is_missing_or_unknown:
        review_flags.append("Missing or unrecognized source")
    if interaction_date is None or (parse_date(lead["last_interaction_date"]) and parse_date(lead["last_interaction_date"]) > reference_date):
        review_flags.append("Missing, invalid, or future last interaction date")
    if size is None:
        review_flags.append("Missing or invalid company size")

    # Missing required data always triggers review. When data is complete, use the score bands.
    if review_flags:
        decision = "review"
    elif final_score >= 5:
        decision = "qualified"
    # elif final_score >= 5:
    #     decision = "review"
    else:
        decision = "rejected"

    rejection_reason = ""
    if decision == "rejected":
        reasons = []
        if intent_score <= 4:
            reasons.append("weak buying-intent signal")
        if recency_score <= 2:
            reasons.append("stale interaction")
        if company_size_score == 6:
            reasons.append("company size outside preferred segment")
        rejection_reason = "; ".join(reasons) if reasons else "weighted score below qualification threshold"

    return {
        **lead,
        "lead_id": f"L{input_index:04d}",
        "input_index": input_index,
        "days_since_interaction": days_since,
        "intent_score": intent_score,
        "recency_score": recency_score,
        "company_size_score": company_size_score,
        "data_quality_score": data_quality_score,
        "raw_weighted_score": round(raw_score, 4),
        "qualification_score": final_score,
        "decision": decision,
        "review_flags": review_flags,
        "rejection_reason": rejection_reason,
        "reasoning": "",
        "outreach_message": "",
        "llm_status": "not_run",
    }


def read_leads(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is empty or has no header row.")
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError("Input CSV is missing required columns: " + ", ".join(missing_columns))
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV contains a header but no leads.")
    return rows


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "leads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "outreach_message": {"type": "string"},
                    },
                    "required": ["lead_id", "reasoning", "outreach_message"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["leads"],
        "additionalProperties": False,
    }


def _api_retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)):
        return True
    return isinstance(exc, APIStatusError) and getattr(exc, "status_code", 0) >= 500
def _get_retry_delay(exc: Exception, attempt: int) -> float:
    """Calculate retry delay, respecting Groq's rate-limit guidance."""

    # Default exponential backoff
    delay = min(30.0, (2 ** (attempt - 1)) + random.uniform(0, 0.5))

    # Try to read the provider's retry-after information
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}

    retry_after = (
        headers.get("retry-after")
        or headers.get("x-ratelimit-reset-requests")
        or headers.get("x-ratelimit-reset-tokens")
    )

    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except (ValueError, TypeError):
            pass

    return min(60.0, delay)

def analyze_batch(client: Any, batch: list[dict[str, Any]], model: str, max_attempts: int = 4) -> dict[str, dict[str, str]]:
    """Make one LLM call for a batch, with bounded retries and output validation."""
    payload = []
    for lead in batch:
        payload.append({
            "lead_id": lead["lead_id"],
            "name": lead["name"],
            "company": lead["company"],
            "company_size": lead["company_size"],
            "industry": lead["industry"],
            "source": lead["source"],
            "last_interaction_date": lead["last_interaction_date"],
            "score": lead["qualification_score"],
            "decision": lead["decision"],
            "factor_scores": {
                "buying_intent": lead["intent_score"],
                "recency": lead["recency_score"],
                "company_size": lead["company_size_score"],
                "data_quality": lead["data_quality_score"],
            },
            "raw_score": lead["raw_weighted_score"],
            "review_flags": lead["review_flags"],
            "rejection_reason": lead["rejection_reason"],
        })

    system_prompt = (
        "You are assisting a lead qualification CLI. For every input lead, return one concise, "
        "evidence-based explanation of the already-computed score and decision. Do not change, "
        "reinterpret, or recommend changing scores or decisions. Use only supplied fields and "
        "rubric signals; do not infer product features, budget, urgency, authority, or industry fit. "
        "For qualified leads only, draft a short, polite first outreach message using available facts. "
        "Do not invent a product name or claim the recipient requested a demo unless the source says "
        "Inbound demo request. If name/company is missing, avoid pretending to know it. For review or "
        "rejected leads, outreach_message must be an empty string. Return exactly one result for every "
        "lead_id, with no duplicates."
    )
    user_prompt = "Analyze this batch and return JSON matching the schema. Input leads:\n" + json.dumps(payload, ensure_ascii=False)

    expected_ids = {lead["lead_id"] for lead in batch}
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={"format": {"type": "json_schema", "name": "lead_batch_analysis", "strict": True, "schema": _schema()}},
            )
            parsed = json.loads(response.output_text)
            entries = parsed.get("leads")
            if not isinstance(entries, list):
                raise ValueError("LLM response did not contain a leads array")
            output: dict[str, dict[str, str]] = {}
            for entry in entries:
                lead_id = entry.get("lead_id")
                if not isinstance(lead_id, str) or lead_id not in expected_ids or lead_id in output:
                    raise ValueError("LLM response contains an unknown or duplicate lead_id")
                if not isinstance(entry.get("reasoning"), str) or not isinstance(entry.get("outreach_message"), str):
                    raise ValueError("LLM response contains invalid field types")
                output[lead_id] = {"reasoning": entry["reasoning"].strip(), "outreach_message": entry["outreach_message"].strip()}
            if set(output) != expected_ids:
                raise ValueError("LLM response omitted one or more leads")
            return output
        except Exception as exc:  # Includes schema/output validation failures.
            last_error = exc
            retryable = _api_retryable(exc) or isinstance(exc, (ValueError, json.JSONDecodeError))
            if not retryable or attempt >= max_attempts:
                break
            delay = _get_retry_delay(exc, attempt)
            logging.warning(
                "LLM batch attempt %s/%s failed (%s); retrying in %.1fs",
                attempt,
                max_attempts,
                type(exc).__name__,
                delay
            )
            time.sleep(delay)
    raise RuntimeError(f"LLM batch failed after {max_attempts} attempt(s): {last_error}") from last_error


def enrich_with_llm(leads: list[dict[str, Any]], batch_size: int, model: str, skip_llm: bool = False) -> list[str]:
    """Enrich leads in batches. Returns batch-level errors for the run summary."""
    errors: list[str] = []
    if skip_llm:
        for lead in leads:
            lead["reasoning"] = build_fallback_reasoning(lead)
            lead["llm_status"] = "skipped"
        return errors

    if OpenAI is None:
        raise RuntimeError("The OpenAI SDK is not installed. Run: pip install -r requirements.txt")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Set it as an environment variable; do not hardcode your key.")

    # Disable SDK auto-retries because this module implements and logs bounded retries itself.
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        max_retries=0,
        timeout=60.0
    )
    for start in range(0, len(leads), batch_size):
        batch = leads[start:start + batch_size]
        try:
            results = analyze_batch(client, batch, model)
            for lead in batch:
                result = results[lead["lead_id"]]
                lead["reasoning"] = result["reasoning"] or build_fallback_reasoning(lead)
                # Only qualified leads can receive an outreach message, regardless of LLM output.
                lead["outreach_message"] = result["outreach_message"] if lead["decision"] == "qualified" else ""
                lead["llm_status"] = "success"
        except Exception as exc:
            message = f"Batch starting at input row {batch[0]['input_index']} failed: {exc}"
            logging.exception(message)
            errors.append(message)
            for lead in batch:
                lead["reasoning"] = build_fallback_reasoning(lead)
                lead["outreach_message"] = ""
                lead["llm_status"] = "failed_fallback"
    return errors


def build_fallback_reasoning(lead: dict[str, Any]) -> str:
    parts = [f"Buying-intent score {lead['intent_score']}/10", f"recency score {lead['recency_score']}/10", f"company-size score {lead['company_size_score']}/10", f"data-quality score {lead['data_quality_score']}/10"]
    if lead["review_flags"]:
        parts.append("manual review required: " + ", ".join(lead["review_flags"]))
    elif lead["decision"] == "qualified":
        parts.append("meets the qualification threshold with all qualification-required fields present")
    elif lead["decision"] == "review":
        parts.append("score falls in the review band")
    else:
        parts.append(lead["rejection_reason"] or "score falls below the qualification threshold")
    return "; ".join(parts) + "."


def rank_leads(leads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Qualified leads first; then final score, raw score, recency, input order."""
    decision_order = {"qualified": 0, "review": 1, "rejected": 2}
    ranked = sorted(
        leads,
        key=lambda lead: (
            decision_order[lead["decision"]],
            -lead["qualification_score"],
            -lead["raw_weighted_score"],
            (lead["days_since_interaction"] if lead["days_since_interaction"] is not None else 10**9),
            lead["input_index"],
        ),
    )
    for rank, lead in enumerate(ranked, start=1):
        lead["priority_rank"] = rank
    return ranked


def build_summary(leads: list[dict[str, Any]], llm_errors: list[str], reference_date: date, batch_size: int) -> dict[str, Any]:
    total = len(leads)
    decisions = Counter(lead["decision"] for lead in leads)
    rejection_reasons = Counter(lead["rejection_reason"] for lead in leads if lead["decision"] == "rejected" and lead["rejection_reason"])
    qualified = [lead for lead in leads if lead["decision"] == "qualified"]
    outreach = [
        {"lead_id": lead["lead_id"], "name": lead["name"], "company": lead["company"], "message": lead["outreach_message"]}
        for lead in qualified if lead["outreach_message"]
    ][:5]
    return {
        "reference_date": reference_date.isoformat(),
        "total_processed": total,
        "qualified_count": decisions["qualified"],
        "qualified_percentage": round(100 * decisions["qualified"] / total, 2) if total else 0,
        "review_count": decisions["review"],
        "rejected_count": decisions["rejected"],
        "common_rejection_reasons": [{"reason": reason, "count": count} for reason, count in rejection_reasons.most_common()],
        "sample_outreach_messages": outreach,
        "batch_size": batch_size,
        "llm_batch_errors": llm_errors,
        "llm_status_counts": dict(Counter(lead["llm_status"] for lead in leads)),
    }


def write_outputs(ranked: list[dict[str, Any]], output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Human-readable CSV: serialize list fields as JSON strings.
    csv_path = output_dir / "lead_results.csv"
    fieldnames = [
        "lead_id", "priority_rank", "name", "company", "company_size", "industry", "source", "last_interaction_date",
        "qualification_score", "raw_weighted_score", "decision", "intent_score", "recency_score", "company_size_score",
        "data_quality_score", "days_since_interaction", "review_flags", "rejection_reason", "reasoning", "outreach_message", "llm_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for lead in ranked:
            row = dict(lead)
            row["review_flags"] = "; ".join(lead["review_flags"])
            writer.writerow(row)

    json_path = output_dir / "lead_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(ranked, handle, indent=2, ensure_ascii=False, default=str)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score, qualify, rank, and enrich inbound leads.")
    parser.add_argument("--input", type=Path, required=True, help="Input leads CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for CSV/JSON reports")
    parser.add_argument("--reference-date", default=date.today().isoformat(), help="Scoring reference date YYYY-MM-DD (defaults to today)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of leads sent in each LLM API call (default: 10)")
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        help="Groq model (default: GROQ_MODEL or openai/gpt-oss-120b)"
    )
    parser.add_argument("--skip-llm", action="store_true", help="Run deterministic scoring only; useful for tests/offline checks")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stderr)])
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    try:
        reference_date = date.fromisoformat(args.reference_date)
        rows = read_leads(args.input)
        scored = [score_lead(row, reference_date, index) for index, row in enumerate(rows, start=1)]
        llm_errors = enrich_with_llm(scored, args.batch_size, args.model, skip_llm=args.skip_llm)
        ranked = rank_leads(scored)
        summary = build_summary(ranked, llm_errors, reference_date, args.batch_size)
        write_outputs(ranked, args.output_dir, summary)
        logging.info("Processed %s leads: %s qualified, %s review, %s rejected", summary["total_processed"], summary["qualified_count"], summary["review_count"], summary["rejected_count"])
        logging.info("Reports written to %s", args.output_dir.resolve())
        if llm_errors:
            logging.warning("One or more LLM batches failed; deterministic results were retained with fallback reasoning.")
        return 0
    except Exception as exc:
        logging.exception("Lead processing failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
