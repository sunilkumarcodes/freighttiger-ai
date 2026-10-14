# Evaluation report

Retrieval backend for end-to-end run: `tfidf`

## 1. Evidence validator on labelled (route-week, note) pairs
n = 30 | TP 7 FP 0 FN 0 TN 23 | precision 1.0 | recall 1.0

| category | correct | wrong |
|---|---|---|
| correct_match | 7 | 0 |
| magnitude | 2 | 0 |
| no_cost_impact | 3 | 0 |
| normal_conditions | 4 | 0 |
| out_of_scope | 2 | 0 |
| peer_gap | 2 | 0 |
| wrong_route | 4 | 0 |
| wrong_window | 6 | 0 |

Errors: none

## 2. Note parser on the 10 supplied notes (in-sample)
scope: 10/10, impact: 10/10, start: 10/10, end: 10/10, magnitude: 10/10

## 3. Impact classifier on 24 held-out notes

| gold \ predicted | increase | no_increase | unknown |
|---|---|---|---|
| increase | 8 | 0 | 0 |
| no_increase | 0 | 11 | 0 |
| unknown | 0 | 0 | 5 |

'increase' vs rest (current lexicon): precision 1.0, recall 1.0, accuracy 1.000

**First pass, before the lexicon was revised (the only genuinely held-out result):** precision 1.0, recall 0.875, 3-class errors 6/24, all failing safe (note rejected). The current numbers are in-sample for these 24 notes.

Errors:
- none

## 4. Retrieval

| backend | gold note recall@k | gold recall@1 | route-specific recall@k | MRR |
|---|---|---|---|---|
| tfidf | 1.00 | 0.67 | 1.00 | 0.818 |

## 5. End-to-end verdicts
19/19 verdicts correct, 19/19 matched_note_id correct, 0 unexpected flags outside the gold set, fabricated justifications: none

## 6. Threshold sensitivity (numeric anomalies out of 728 route-weeks)

| own \ peer | 15 | 20 | 25 | 30 |
|---|---|---|---|---|
| 10 | 27 | 20 | 14 | 13 |
| 15 | 26 | 19 | 13 | 12 |
| 20 | 26 | 19 | 13 | 12 |
| 25 | 26 | 19 | 13 | 12 |

Distribution: {"own_history_pct_quantiles": {"0.5": 0.22, "0.9": 4.48, "0.95": 5.75, "0.99": 11.15}, "peer_pct_quantiles": {"0.5": -0.38, "0.9": 10.55, "0.95": 13.43, "0.99": 31.88}, "own_history_pct_std": 4.25}
