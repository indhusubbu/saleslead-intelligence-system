# Lead Intelligence System

A Python CLI prototype that automates inbound lead qualification,
scoring, prioritization, and personalized outreach generation.

The system uses deterministic business rules to calculate lead scores
and assign qualification decisions. The Groq LLM API is used in batches
to generate lead-specific explanations and personalized first-contact
messages. The LLM cannot modify or override the deterministic scores or
decisions.

## 1. Features

- Ingests lead data from CSV files.
- Calculates weighted qualification scores from 1–10.
- Classifies leads as `qualified`, `rejected`, or `review`.
- Applies mandatory human-review rules for incomplete critical
  information.
- Ranks qualified leads by priority.
- Uses the Groq LLM API for batch-based explanations and personalized
  outreach messages.
- Handles API failures, rate limits, and invalid LLM responses through
  retries and fallback reasoning.
- Generates CSV and JSON reports with aggregate statistics.
- Supports processing 100+ leads per run without manual intervention.

## 2. Requirements

- Python 3.10 or higher.
- Groq API key.
- Internet connectivity for LLM-powered analysis.

The project uses the OpenAI-compatible Python SDK to communicate with
the Groq API.

## 3. Installation and Execution

### Step 1: Create a virtual environment

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

``` powershell
pip install -r requirements.txt
```

### Step 3: Configure Groq API credentials

Set the API key and model in PowerShell:

``` powershell
$env:GROQ_API_KEY="your-groq-api-key"
$env:GROQ_MODEL="openai/gpt-oss-120b"
```

Never hardcode API credentials or commit real API keys to source
control.

The `.env.example` file provides a reference template for the required
configuration. The application does not automatically load `.env` files;
configure the required variables in the execution environment.

### Step 4: Run the 30-lead sample dataset

``` powershell
python lead_intelligence.py --input test_30_leads.csv --reference-date 2024-02-01 --batch-size 10 --output-dir test_30_leads_output
```

### Step 5: Run the 100-lead scale test

``` powershell
The project was also evaluated using a 100-lead dataset supplied as part of the IIT FDE programme. That dataset is not included in this repository. The fictional 30-record sample is provided so that the core application can be tested independently.
```

### Optional: Run without LLM access

The application also supports scoring-only execution without calling the
LLM API:

``` powershell
python lead_intelligence.py --input test_30_leads.csv --reference-date 2024-02-01 --skip-llm --output-dir test_30_leads_output
```

The batch size defaults to 10. For 100 leads with a batch size of 10,
the application processes approximately 10 batches. Additional API calls
may occur if retries are required.

The model can be configured through the `GROQ_MODEL` environment
variable or the `--model` command-line argument.

## 4. Input Data

The application accepts CSV files containing the following fields:

| Field                   | Description                    |
|-------------------------|--------------------------------|
| `name`                  | Lead contact name              |
| `company`               | Company name                   |
| `company_size`          | Number of employees            |
| `industry`              | Industry or business sector    |
| `source`                | Lead acquisition source        |
| `last_interaction_date` | Date of the latest interaction |

The 30-lead dataset contains diverse leads across multiple industries,
company sizes, and acquisition sources, including high-intent,
low-intent, and ambiguous cases.

## 5. Lead Scoring Rubric

Each lead receives a weighted qualification score based on four factors:
buying intent, interaction recency, company size, and data quality.

| Factor              |   Weight |
|---------------------|---------:|
| Buying intent       |      40% |
| Interaction recency |      30% |
| Company size        |      20% |
| Data quality        |      10% |
| **Total**           | **100%** |

### 5.1 Buying Intent

Buying intent represents the level of interest demonstrated by the lead
based on the acquisition source.

| Lead source               | Score |
|---------------------------|------:|
| Inbound demo request      |    10 |
| Referral                  |     9 |
| Content download          |     7 |
| Webinar attendee          |     6 |
| Sales call                |     5 |
| LinkedIn outreach         |     4 |
| Missing or unknown source |     1 |

### 5.2 Interaction Recency

Recent interactions indicate stronger engagement and higher potential
for timely follow-up.

The reference date is configurable. The execution examples use February
1, 2024.

| Days since interaction  | Score |
|-------------------------|------:|
| 0–7 days                |    10 |
| 8–30 days               |     8 |
| 31–90 days              |     5 |
| More than 90 days       |     2 |
| Missing or invalid date |     1 |

### 5.3 Company Size

Company size is used as an assumed indicator of target-market fit.

| Employee count | Score |
|----------------|------:|
| 0–500          |    10 |
| Above 500      |     6 |
| Unknown        |     5 |

### 5.4 Data Quality

Data quality starts at 10. Penalties are applied cumulatively for
missing information.

| Missing information | Penalty |
|---------------------|--------:|
| Contact name        |      -5 |
| Lead source         |      -4 |
| Interaction date    |      -3 |
| Company size        |      -3 |
| Company name        |       0 |
| Industry            |       0 |

The minimum data quality score is 1.

### 5.5 Final Score Calculation

The final qualification score is calculated using the weighted sum of
the four factors:

**Weighted score = (Buying Intent × 0.40) + (Recency × 0.30) + (Company
Size × 0.20) + (Data Quality × 0.10)**

The resulting score is rounded to the nearest whole number using half-up
rounding and constrained to the range of 1–10.

The final score is used by the qualification decision policy described
in Section 7. Mandatory human-review conditions take precedence over the
numerical score.

## 6. Business Assumptions and Design Decisions

The lead qualification framework is based on the following business
assumptions and design decisions.

### Business assumptions

1.  **Buying intent is the strongest indicator:** Leads showing direct
    interest, such as demo requests and referrals, receive higher intent
    scores than passive engagement or outbound prospecting.
2.  **Recent interactions indicate stronger engagement:** Leads
    contacted recently are generally considered more actionable than
    leads with older interactions.
3.  **Company size is used as a fit indicator:** Companies with up to
    500 employees receive a higher score in this prototype. This is an
    assumed target-market preference, not a universal measure of
    business value.
4.  **Data completeness affects confidence:** Missing critical lead
    information reduces the data quality score and may prevent automatic
    qualification.
5.  **Incomplete information requires human judgment:** Leads missing
    critical fields are sent for review rather than being automatically
    accepted or rejected.

### Design decisions

1.  **Deterministic scoring:** Python business rules calculate all
    weighted scores and qualification decisions. This ensures consistent
    and reproducible results.
2.  **LLM as an enrichment layer:** Groq generates explanations and
    personalized outreach messages but cannot modify scores, decisions,
    or priority rankings.
3.  **Human review takes precedence:** Any mandatory review condition
    overrides the numerical qualification threshold.
4.  **Configurable reference date:** A reference date is used for
    recency calculations so results can be reproduced and tested against
    historical datasets.
5.  **Batch processing:** Leads are sent to the LLM in configurable
    batches to reduce individual API calls and support larger datasets.
6.  **Fallback processing:** If LLM enrichment fails after retries,
    deterministic scoring results are preserved and fallback reasoning
    is used.

## 7. Qualification and Human Review

The qualification decision is based on the final score and mandatory
review flags:

- Score of 5–10 without mandatory review flags: `qualified`.
- Score of 1–4 without mandatory review flags: `rejected`.
- Any mandatory review flag: `review`, regardless of score.

Critical information includes the contact name, lead source, last
interaction date, and company size. Missing or invalid critical
information, or an unrecognized lead source, can require human review.
An unknown company size is treated as incomplete information and
requires review.

A lead assigned `review` is not automatically treated as qualified or
rejected. It is held for a person to inspect and resolve the missing or
ambiguous information.

## 8. Priority Ranking

Qualified leads are ranked for follow-up using the following order:

1.  Qualified leads are placed ahead of leads that are rejected or
    awaiting review.
2.  Within the qualified group, higher final scores rank first.
3.  If final scores are equal, higher unrounded weighted scores rank
    first.
4.  If scores are still equal, more recent interactions rank first.
5.  Remaining ties retain a deterministic order based on the input
    sequence.

This ordering makes the priority queue reproducible while giving
preference to stronger-scoring and more recently engaged leads.

## 9. LLM Integration and Error Handling

The Groq LLM API is used to generate lead-specific explanations and
personalized first-contact messages. Leads are submitted in configurable
batches rather than making a separate API request for every lead.

The deterministic scoring and decision logic runs independently of the
LLM. LLM responses are used for enrichment only and cannot change the
score, qualification decision, or priority rank.

The application includes retry handling for API failures and invalid
responses. If a batch cannot be enriched successfully after the
configured retries, the application preserves the deterministic results
and uses fallback reasoning so that processing can continue.

LLM output quality and availability depend on the selected model and API
service. A successful lead-processing run does not necessarily mean
every lead received a successful LLM-generated message.

## 10. Output Reports

The application generates per-lead and aggregate reports in the selected
output directory:

- `lead_results.csv` — tabular per-lead results.
- `lead_results.json` — per-lead results in JSON format.
- `summary.json` — aggregate processing statistics.

Per-lead results include the qualification score, decision, reasoning,
and priority information. The reports also support inspection of
qualified, rejected, and review leads. LLM-generated explanations and
outreach content are included where available; fallback content is used
when enrichment fails.

## 11. Limitations

- The scoring weights, company-size preference, and qualification
  threshold are prototype assumptions. They have not been calibrated
  against historical conversion data.
- LLM-generated explanations and outreach messages may vary between runs
  and should be reviewed before being sent to prospects.
- LLM enrichment requires a working API connection and valid
  credentials. Deterministic scoring can be run without LLM access using
  `--skip-llm`.
- The prototype does not connect to a CRM or independently verify lead
  information.

Business owners should validate the scoring assumptions against actual
sales outcomes before production deployment.

## 12. Testing

Run the automated unit tests from the project root:

``` powershell
python -m unittest discover -s tests -v
```

The latest recorded test run completed with **14 tests passed**.

## 13. Sample Execution Results

The following results were recorded during local validation of the Lead
Intelligence System.

| Test                         |              Result |
|------------------------------|--------------------:|
| Sample dataset               |  30 leads processed |
| Scale dataset                | 100 leads processed |
| Unit tests                   |           14 passed |
| Latest 100-lead LLM success  |                  90 |
| Latest 100-lead LLM fallback |                  10 |

**Note:** The 100-lead run successfully processed all 100 records. LLM
enrichment succeeded for 90 leads, while the remaining 10 used the
fallback mechanism. These results represent a specific test run and may
vary depending on API availability and model responses.
