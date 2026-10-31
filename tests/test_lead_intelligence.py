import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lead_intelligence import score_lead, rank_leads, read_leads, round_score  # noqa: E402

REFERENCE = date(2024, 2, 1)
BASE = {
    "name": "Avery Example",
    "company": "Example Co",
    "company_size": "200",
    "industry": "SaaS",
    "source": "Inbound demo request",
    "last_interaction_date": "2024-01-31",
}


class LeadScoringTests(unittest.TestCase):
    def test_baseline_scores_ten_and_qualifies(self):
        lead = score_lead(BASE, REFERENCE, 1)
        self.assertEqual(lead["qualification_score"], 10)
        self.assertEqual(lead["decision"], "qualified")
        self.assertEqual(lead["data_quality_score"], 10)

    def test_missing_contact_name_reduces_quality_and_forces_review(self):
        row = {**BASE, "name": ""}
        lead = score_lead(row, REFERENCE, 1)
        self.assertEqual(lead["data_quality_score"], 5)
        self.assertEqual(lead["decision"], "review")
        self.assertEqual(lead["qualification_score"], 10)  # high score does not bypass review

    def test_missing_source_uses_intent_one_and_forces_review(self):
        row = {**BASE, "source": ""}
        lead = score_lead(row, REFERENCE, 1)
        self.assertEqual(lead["intent_score"], 1)
        self.assertEqual(lead["data_quality_score"], 6)
        self.assertEqual(lead["decision"], "review")

    def test_missing_industry_has_no_penalty(self):
        row = {**BASE, "industry": ""}
        lead = score_lead(row, REFERENCE, 1)
        self.assertEqual(lead["data_quality_score"], 10)
        self.assertEqual(lead["decision"], "qualified")

    def test_missing_company_size_uses_neutral_five_and_review(self):
        row = {**BASE, "company_size": ""}
        lead = score_lead(row, REFERENCE, 1)
        self.assertEqual(lead["company_size_score"], 5)
        self.assertEqual(lead["data_quality_score"], 7)
        self.assertEqual(lead["qualification_score"], 9)
        self.assertEqual(lead["decision"], "review")

    def test_missing_date_forces_review(self):
        lead = score_lead({**BASE, "last_interaction_date": ""}, REFERENCE, 1)
        self.assertEqual(lead["recency_score"], 1)
        self.assertEqual(lead["data_quality_score"], 7)
        self.assertEqual(lead["decision"], "review")

    def test_cumulative_penalty_floors_at_one(self):
        row = {**BASE, "name": "", "source": "", "last_interaction_date": "", "company_size": ""}
        lead = score_lead(row, REFERENCE, 1)
        self.assertEqual(lead["data_quality_score"], 1)
        self.assertEqual(lead["decision"], "review")

    def test_recency_boundaries(self):
        cases = [(0, 10), (7, 10), (8, 8), (30, 8), (31, 5), (90, 5), (91, 2)]
        from lead_intelligence import score_recency
        for days, expected in cases:
            with self.subTest(days=days):
                self.assertEqual(score_recency(days), expected)

    def test_rounding_half_up(self):
        self.assertEqual(round_score(8.5), 9)
        self.assertEqual(round_score(7.5), 8)

    def test_ranking_qualified_first_then_scores(self):
        leads = [
            {"decision": "review", "qualification_score": 10, "raw_weighted_score": 9.7, "days_since_interaction": 0, "input_index": 1},
            {"decision": "qualified", "qualification_score": 9, "raw_weighted_score": 8.9, "days_since_interaction": 10, "input_index": 2},
            {"decision": "qualified", "qualification_score": 9, "raw_weighted_score": 9.4, "days_since_interaction": 20, "input_index": 3},
        ]
        ranked = rank_leads(leads)
        self.assertEqual([x["input_index"] for x in ranked], [3, 2, 1])
        self.assertEqual([x["priority_rank"] for x in ranked], [1, 2, 3])

    def test_sample_csv_has_at_least_30_leads(self):
        rows = read_leads(ROOT / "test_30_leads.csv")
        self.assertGreaterEqual(len(rows), 30)

    def test_unknown_source_is_review_and_penalized(self):
        lead = score_lead({**BASE, "source": "Unknown channel"}, REFERENCE, 1)
        self.assertEqual(lead["intent_score"], 1)
        self.assertEqual(lead["data_quality_score"], 6)
        self.assertEqual(lead["decision"], "review")

    def test_recency_tiebreaker_puts_more_recent_first(self):
        leads = [
            {"decision": "qualified", "qualification_score": 9, "raw_weighted_score": 8.8, "days_since_interaction": 20, "input_index": 1},
            {"decision": "qualified", "qualification_score": 9, "raw_weighted_score": 8.8, "days_since_interaction": 2, "input_index": 2},
        ]
        ranked = rank_leads(leads)
        self.assertEqual([x["input_index"] for x in ranked], [2, 1])

    def test_can_score_at_least_100_leads_without_manual_steps(self):
        leads = [score_lead({**BASE, "name": f"Lead {i}"}, REFERENCE, i) for i in range(1, 101)]
        self.assertEqual(len(leads), 100)
        self.assertTrue(all(1 <= lead["qualification_score"] <= 10 for lead in leads))


if __name__ == "__main__":
    unittest.main(verbosity=2)
