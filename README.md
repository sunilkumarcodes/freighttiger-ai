# FreightTiger: freight cost anomaly assistant

This project looks at shipment records, works out each route's weekly cost per tonne-km, and
flags weeks where the cost is rising in a way that looks out of the ordinary. For every flagged
week it searches the context notes for a reason. A week only counts as justified if a note
really applies to that route, that week, and actually describes a cost increase. Otherwise the
week is left flagged for someone to review.

Result on the supplied data: 728 route-weeks, of which 16 are flagged (`Yes`), 3 are justified
(`No (justified)`) and 709 are normal (`No`).

## Running it

Python 3.11 or newer is needed (pandas 3).

```
pip install -r requirements.txt
python run.py                              # writes outputs/final_output.csv and the logs
pytest -q                                  # 129 tests
python scripts/evaluate.py                 # scores against the labelled sets in eval/
python scripts/reproducibility_check.py    # 3 separate runs, compares hashes
streamlit run app.py                       # dashboard
python scripts/finalize.py                 # everything above with the semantic model, then updates this README
```

The first run downloads the sentence-transformer model (about 90 MB). After that everything
runs offline and no API keys are needed.

## How it works

```
shipments -> clean/validate -> weekly cost per route -> compare with history and peers -> flag
                                                                                      |
notes -> parse each note (route, dates, cost impact) -----> check candidate notes <- retrieve notes
                                                                    |
                                                  verdict + short reason -> CSV
```

The code is in `src/`:

- `ingestion.py` loads the CSVs and drops bad rows: non-numeric values, zero or negative
  tonnes, km or cost, bad dates, unknown route types, duplicate ids. Every dropped row is
  counted in a data quality report.
- `metrics.py` does the weekly numbers.
- `anomaly_detection.py` applies the flag rule.
- `note_parser.py` turns each note into structured fields (which routes, which dates, does it
  say costs went up, by how much). It also records the phrase each field came from.
- `retrieval.py` embeds the notes and finds the closest ones for each flagged week.
- `evidence_validation.py` decides whether a retrieved note is acceptable.
- `explanations.py` writes the reason text.
- `pipeline.py` ties it together. `app.py` is the Streamlit dashboard.

## The numbers

Weeks run Monday to Sunday and `week_of` is the Monday.

Weekly cost per tonne-km for a route is the total cost of that week's shipments divided by
their total tonne-km:

    C = sum(freight_cost) / sum(quantity_tonnes * distance_km)

I checked this against the brief's three sample rows. This formula reproduces all 9 numbers in
them. Averaging each shipment's own ratio instead gives 4.16 and +35.2% where the sample has
4.17 and +35.5%, so that reading is ruled out.

**vs own history** compares the week with the average of the route's weekly values from the
previous 8 weeks, never including the current week or anything after it. For the first 8 weeks
of each route (2024-01-01 to 2024-02-19) fewer than 8 earlier weeks exist, so I average the
ones that do, without padding. The very first week has nothing to compare against and shows
`n/a`. The reason text says "only n prior weeks available" in these cases. None of these
early weeks was flagged. Weeks with no shipments are simply missing from the series, not
treated as zero. In this data every route has all 104 weeks, so that never comes up.

**vs similar routes** compares the week with the plain average of the other routes of the same
route_type in that same week, leaving the route itself out.

Both are shown as percentage differences. Decisions use the unrounded values. The CSV shows
cost to 2 decimals and percentages to 1.

To check the arithmetic I wrote a second, independent version in plain Python (csv and
datetime, no pandas). It agrees with the pipeline on all 728 route-weeks.

## When a week gets flagged

A week is a numeric anomaly when its cost is above its own history average (the brief says
"rising") and it is either at least 15% above its own history or at least 20% above similar
routes.

The brief doesn't give thresholds, so these are my choice. They're set in `src/config.py`.
Why these values:

- Week-to-week changes vs history are small: median 0.2%, standard deviation about 4.3%,
  99th percentile 11.2%. 15% is well outside normal noise, and the diesel price rise
  (+4 to 9% on every route) stays under it, which is right.
- The peer comparison has a built-in offset: short routes cost more per tonne-km. Mumbai-Pune
  normally sits about 9% above its only Short peer, Delhi-Jaipur. A 20% threshold leaves room
  for that, but still catches Mumbai-Pune drifting to 20 to 39% above its peer from mid-2025.
  Its own-history comparison can't catch that slow creep, because the 8-week average rises
  along with it.
- Sensitivity: with the peer threshold at 20%, own-history thresholds of 10/15/20/25% give
  20/19/19/19 anomalies. Lowering the peer threshold to 15% gives 26 to 27. So the peer
  threshold is the one that matters, and it's what I'd tune with real reviewer feedback.
- The rule reproduces the flags in all three sample rows, including Mumbai-Pune 2025-09-15
  (+9.2% vs history, +23.6% vs peers, flagged).

One borderline case: Mumbai-Pune 2025-11-10 is 21% above its peer but 0.09% below its own
average, so it doesn't count as rising and isn't flagged. The weeks around it are flagged.

## Finding and checking reasons

For each flagged week I build a query like "Why did freight transport costs rise on the
Chennai-Bangalore route in the week of 03 March 2025?". I embed it with
`all-MiniLM-L6-v2` and take the 5 most similar notes. A TF-IDF retriever is included as a
fallback and as a baseline to compare against.

Retrieval only suggests notes. A note justifies a week only if all of these checks pass:

1. **Route.** The note is for this route (either direction) or for all routes. N004 says its
   routes aren't in the dataset, so it never applies.
2. **Dates.** The note's period overlaps the week. An explicit range like "Feb 24 to Mar 8" is
   used as written. An open-ended note like N003 ("starting this week") is applied for 8
   weeks, after which the change is part of the 8-week average anyway. A note with no stated
   duration only covers the week it's dated. A note never applies before its own date unless its
   text gives an explicit earlier start ("flooding from Feb 20 to Mar 8", reported on Feb 24).
3. **Cause.** The note says costs went up. "Costs were not significantly affected" (N005),
   "absorbed without a rate change" (N010), and notes about normal or improved conditions
   (N006, N007, N008, N009) don't count. Neither do negated or hypothetical statements
   ("no surcharge was applied", "rates could rise").
4. **Size.** If the note gives a size, the rise can't be much bigger. N003 says roughly 5 to
   7%, so it can't explain more than about 10%.
5. **Peer gap.** A nationwide cause raises the other routes too, so it can't explain one route
   being far above its peers.

If no note passes, the week stays flagged and the reason names the closest note and why it
didn't qualify. Explanations are filled-in templates: computed numbers plus a direct quote from
the note. Nothing is generated freely, so there's nothing to hallucinate.

What this gives on the supplied notes:

| Route and week | Verdict |
|---|---|
| Ahmedabad-Mumbai 2025-01-20 | justified by N002 (festival surcharge) |
| Chennai-Bangalore 2025-02-24 and 03-03 | justified by N001 (flooding, Feb 24 to Mar 8) |
| Chennai-Bangalore 2025-03-10 and 03-17 | flagged: the flood ended Mar 8 and N009 says the route was back to normal |
| Delhi-Jaipur 2024-11-11 and 11-18 | flagged: no note (N007 is from May and is about better roads) |
| Mumbai-Pune, 12 weeks from 2025-06-23 | flagged: steady rise vs peers; N003, N006 and N010 don't explain it |

I didn't put an LLM in the decision path. The verdict has to be repeatable and checkable, and
the brief treats a made-up reason as worse than no reason. Where an LLM would fit is drafting
the structured fields for new notes, with a person checking them.

## Output

`outputs/final_output.csv` has the same columns, in the same order, as the sample. It has one
row per route-week, so every weekly number can be checked, not just the flagged ones.
`outputs/flagged_only.csv` has just the 19 flagged and justified rows. `matched_note_id` is
filled only for justified rows. Text fields are properly quoted (the sample's third row has an
unquoted comma, which breaks it as CSV).

`outputs/route_week_audit.csv` has the full-precision numbers, the retrieved notes with scores,
and the result of every check for every row.

## How I checked it

- **Tests:** `pytest -q` runs 129 tests. They cover:
  - week boundaries and the cost formula
  - the history window, including the case with fewer than 8 weeks
  - leaving the route out of its peer average
  - bad input rows
  - the flag thresholds
  - each kind of note rejection
  - CSV quoting
  - the sample rows
  - the independent recalculation
- **Gold labels:** before writing the checker, I wrote down the verdict I expected for each of
  the 19 anomalies by reading the notes (`eval/gold_verdicts.csv`). The system matches all 19
  verdicts and all 19 note ids. No week was justified by the wrong note.
- **Note pairs:** 30 labelled route-week/note pairs covering every rejection type
  (`eval/labeled_note_pairs.csv`). All 30 are correct.
- **New wording:** 24 notes written in phrasing not used in the dataset
  (`eval/heldout_impact_notes.csv`). The first version of the parser got 18 of 24. All 6
  mistakes were on the safe side: a note was rejected, never wrongly accepted. I then fixed the
  general patterns, so the current 24/24 is no longer an unseen test. 18/24 is the honest
  number.
- **Adversarial notes:** I wrote another 16 notes designed to trick the parser. One exposed a
  real bug: "Costs were not increased; any surcharge was waived" was read as an increase
  because it contains "surcharge". Negation, cancellation and speculation are now handled at
  the clause level. These cases are regression tests. The supplied notes never triggered the
  bug, so the output didn't change.
- **Second audit:** a later review found one more unsafe case: a cancelling verb *before* the
  cue ("transporters removed the festival surcharge", "scrapped the toll hike") was read as an
  increase. These are now rejected. The same audit made three fixes that only affect notes
  the parser used to reject wrongly, so none can let a bad note through: "but"/"however" now
  end a clause, so "no flooding, but rates rose" counts as an increase; "unaffected" only
  negates when a cost word is next to it; and an "All Routes" note whose text names a
  corridor ("toll hike on the Mumbai-Pune expressway") is limited to that corridor. All are
  regression tests. Outputs and evaluation scores are unchanged.
- **Retrieval depth:** validating all 10 notes instead of the top 5 gives exactly the same
  verdicts, so the ranking can't change an outcome on this data.

All of these label sets were written by me, so perfect scores mean the system agrees with a
careful reading of these notes. They are not a general accuracy figure.

<!-- RETRIEVAL:START -->
Retrieval quality on the 19 flagged-week queries (is the right note in the top 5?):

| Retriever | correct note in top 5 | correct note ranked 1st | route's own notes in top 5 | MRR |
|---|---|---|---|---|
| sentence-transformer | not measured yet (run `python scripts/finalize.py` with internet) | | | |
| TF-IDF (baseline) | 1.00 | 0.67 | 1.00 | 0.818 |
<!-- RETRIEVAL:END -->

## Reproducibility

`scripts/reproducibility_check.py` runs the whole pipeline 3 times in separate processes. It
compares SHA-256 hashes and every cell. All three runs were byte-identical, including the
reason text (`outputs/reproducibility_report.txt`). Nothing is random: there is no LLM, sorts
are stable, seeds are fixed, and similarity scores are rounded before ranking, with ties broken
by note id. Input hashes ignore line endings, since git converts them on Windows, and
`.gitattributes` keeps them stable.

## Cost

`outputs/cost_report.json`: 0 LLM calls, 0 input and output tokens, $0. Embeddings are computed
locally: 10 notes plus one query per flagged week (19), in a single batch.

<!-- BACKEND:START -->
Note on the committed outputs: they were produced with the TF-IDF retriever, because the environment they were generated in could not download the sentence-transformer weights (huggingface.co was blocked). The verdicts don't depend on the retriever (see above). Running `python scripts/finalize.py` with internet access regenerates everything with the semantic model and updates this paragraph and the retrieval table.
<!-- BACKEND:END -->

## Limitations

- The note parser is rule-based. It is easy to audit and fails safe, but new phrasing for a
  real cost increase can be missed (the flag then stays, which is the safe outcome). A local
  NLI model as a second opinion would help.
- Notes with no stated duration only cover one week. A longer festival would be under-excused.
  That's on purpose: a reviewer sees it instead of it being excused silently.
- Short and Long routes each have only one peer, so the peer comparison is really a comparison
  with one other route. With more routes I'd use a median, or compare each route's gap with
  its usual gap.
- A slow, permanent price shift gets absorbed into the 8-week average (the Mumbai-Pune
  2025-11-10 case). A drift detector such as CUSUM would complement it.
- Route matching uses the city names in `applies_to`. Real notes would need aliases and
  highway names.
- Scale: the weekly maths is a group-by plus a rolling window, and ran on 294k shipments in
  2 seconds. Retrieval only runs for flagged weeks (about 3% here). With thousands of notes I'd
  move them to a vector store and filter by route and date first.

  platform link: https://freighttiger-ai.streamlit.app/
