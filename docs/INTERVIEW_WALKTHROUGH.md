# 10-minute walkthrough + likely questions

## Script (≈10 min)

**0:00 – Business problem (1 min).** Shippers pay per trip, and cost per tonne-km drifts.
Some drift is legitimate (floods, fuel, festivals); some is not. A cost team wants two lists:
increases that are already explained, and increases a human should challenge. In this setting
a wrong excuse is worse than no excuse: it hides overbilling. So the system is designed to say
"unexplained" whenever it can't prove otherwise.

**1:00 – The metric (1 min).** Cost per tonne-km normalises for load and distance. Weekly, it
is Σcost / Σ(tonnes·km), a ratio of sums, so a tiny shipment can't swing the week. The mean of
per-shipment ratios would give a 1-tonne shipment the same weight as a 20-tonne one. Weeks run
Monday–Sunday.

**2:00 – Baselines (1.5 min).**
* **Own history:** the mean of the 8 calendar weeks strictly before this week. Nothing is
  padded, and early weeks use whatever is available. The test that changes a future week and
  checks past baselines are unchanged is the look-ahead proof.
* **Peers:** the unweighted mean of the *other* same-type routes in the same week
  (leave-one-out).
* **Proof:** the brief's three sample rows are reproduced exactly, and an independent
  pure-Python implementation matches all 728 rows.
* **Why both:** own history catches spikes (Delhi-Jaipur's November 2024 jump), and peers catch
  slow creep that the trailing average absorbs (Mumbai-Pune drifting from +9% to +39% above its
  peer while staying only +3–9% above its own average).

**3:30 – Architecture (1 min).** Show the README diagram. The key line is: *retrieval
proposes, deterministic validation decides.* The three outcomes are stored separately:
numeric anomaly, evidence, and verdict.

**4:30 – Retrieval and validation (1.5 min).**
* A local MiniLM embeds the notes. Each anomaly becomes a question, and cosine similarity
  ranks the notes (top-5).
* Each candidate then must pass five checks: scope, window, causal, magnitude, and peer gap.
* Open `route_week_audit.csv` or the dashboard's *Anomaly explanations* tab to show every check
  with its reason.

**6:00 – Demo: valid justification (1 min).** Chennai-Bangalore, week of 2025-02-24: +31.5%
vs history, N001 flood (Feb 24 – Mar 8). All checks pass, so `No (justified)`, N001, with a
verbatim quote of the note.

**7:00 – Demo: the trap (1 min).** Same route, 2025-03-10: still +22%. N001 is retrieved but
its window ended Mar 8. N009 ("returned to normal") is retrieved, but it states no cost
increase. Result: `Yes`, unexplained. Then Mumbai-Pune 2025-10-27: N010 sits on the same date
and is nationwide, but says costs were absorbed, so it is rejected. N003 diesel is real, but
it's nationwide, so it can't explain a gap vs peers.

**8:00 – Evidence (1 min).**
* 111 tests pass, including adversarial note phrasings found in a self-audit.
* Eval: 19/19 verdicts against gold labels written before coding, 30/30 labelled pairs, 0
  fabricated justifications.
* An honest weakness: the held-out phrasing probe scored 18/24 on the first pass, and every
  error failed safe.
* Three separate-process runs are byte-identical (SHA-256).
* 0 LLM calls, $0.

**9:00 – Limits and next steps (1 min).** The lexicon generalises imperfectly (add NLI as a
second opinion). Duration is conservative. Peer groups are tiny. There's no level-shift
detector (Mumbai-Pune 2025-11-10). Scaling means a SQL window function for metrics plus a
vector DB with a metadata pre-filter.

---

## Likely questions

**1. Why RAG instead of fine-tuning?** The knowledge is a changing set of notes, not a skill. A
retrieval index updates by adding a note; fine-tuning would need retraining and still couldn't
cite a source. Ten notes and no labels also make fine-tuning impossible to do responsibly.
Retrieval gives a citable `note_id`, which is what the grader checks.

**2. Why use an LLM at all?** I don't, in the decision path. The verdict must be reproducible
and auditable, and an LLM adds variance and hallucination risk for no accuracy gain on 10
notes. The "AI" is the semantic retrieval model. An LLM would be justified offline for drafting
structured fields of *new* notes for human review, or for wording answers in the Q&A tab,
constrained to the computed rows. Even then it would never set `flagged` or `matched_note_id`.

**3. What prevents hallucinations?**
* A justification needs a note that passes five deterministic checks.
* The reason text is a template: computed numbers plus a *verbatim* quote of the note.
* `matched_note_id` comes from the validator, not from text generation.
* No supporting note means `Yes`, unexplained.

**4. How do you prevent data leakage / look-ahead?** History uses only weeks strictly before
*t*, and the test proves that changing a future week doesn't move past baselines. A note can't
apply before its own date. The peer baseline uses only the same week (a cross-section, not the
future).

**5. Why these thresholds?**
* 15% own history is about 3.5σ of weekly ΔH (σ = 4.25 pp; 99th percentile 11.2%).
* 20% vs peers is about 10 pp above Mumbai-Pune's structural premium (+9.4%).
* The sensitivity grid shows the own-history threshold barely matters, while the peer
  threshold does (19 vs 26 anomalies at 20% vs 15%).
* The rule also reproduces all three sample rows.
* They're configurable, and with real reviewer labels I'd tune them for precision at a fixed
  review budget.

**6. What if no relevant note exists?** The verdict is `Yes`, `matched_note_id` is blank, and
the reason states the numbers. If a candidate covered the route, the reason also names the
closest note and why it failed. Example: Delhi-Jaipur 2024-11-11, where N007 is about better
roads and is dated May.

**7. What if the embedding model retrieves the wrong note?** It can't produce a wrong
justification: the validator rejects it. The risk is the opposite one, where retrieval *misses*
the right note and we report "unexplained". That fails safe, and we measure it: gold note
recall@5 = 1.0. We also ran with k = all notes and got identical verdicts, so on this data
retrieval rank cannot change an outcome.

**8. How do you prove reproducibility?** Three runs in separate processes, with SHA-256 of the
output and the audit file (byte-identical), plus a cell-by-cell diff including `reason`.
Determinism comes from no sampling, fixed seeds, stable sorts, and rounded similarity scores
with note_id tie-breaks.

**9. How does it scale to millions of shipments?**
* Weekly aggregation is one group-by. History and peers are window functions:
  `AVG(c) OVER (PARTITION BY route ORDER BY week RANGE BETWEEN 8 PRECEDING AND 1 PRECEDING)`,
  run in a warehouse, Polars or DuckDB.
* Retrieval runs only on anomalies (2.6% of route-weeks here), batched. Notes go into
  pgvector/FAISS with a metadata pre-filter (route ∈ {route, ALL}, date window) before vector
  search.
* Cost grows with the number of anomalies, not shipments.

**10. How would you monitor it in production?**
* Data quality: drop rates by reason, and schema drift.
* Weekly anomaly rate per route type.
* Share of anomalies that are justified vs unexplained.
* Retrieval score distributions.
* How often each validation check fails.
* The reviewer override rate, which is the key label source for precision and recall. Alert
  when the justified rate jumps, since that might mean an over-broad note.

**11. Why is Chennai-Bangalore 2025-03-10 flagged when it's obviously flood aftermath?** It may
well be. But N001 explicitly says roads were passable by Mar 9, and N009 says the route
returned to normal. No note claims lingering costs. Excusing it would be inventing a reason, so
a human reviews it. A note saying "rates stayed elevated until Mar 21" would justify it
automatically.

**12. Why not justify Mumbai-Delhi 2025-05-12 (+11.6%) with the diesel note?** It isn't flagged
in the first place (it's under 15% and not anomalous vs peers). If it were, N003 states ~5–7%,
so it can't excuse +11.6%. The magnitude check (upper bound + 3 pp) exists so that a small real
effect doesn't cover a larger unrelated spike.

**13. Why is the output 728 rows rather than just flagged rows?** The brief grades the numeric
fields. Emitting every route-week makes every number checkable, and `flagged = No` is a valid
value in the template. `flagged_only.csv` gives the 19-row view.

**14. What's the weakest part?** The cue lexicon. On 24 held-out phrasings it got 18 right on
the first pass. All 6 errors failed safe, but one was a missed true justification ("toll
charges were hiked"). I fixed the general patterns and say openly that the post-fix 24/24 is no
longer held out. The next step is NLI as a second opinion, with disagreement treated as
unexplained.

**15. How did you handle the malformed sample CSV?** Its third row has an unquoted comma inside
`reason`. I took the header as the contract and wrote RFC-4180 CSV: quote fields that contain
commas, quotes or newlines, and double embedded quotes. Tests re-parse the file with Python's
`csv` module and check that every row has 8 fields and round-trips exactly.

**16. How do you know your cost formula is the intended one?** I recomputed the brief's 9
sample numbers under three formulas. Only Σcost/Σ(tonnes·km) matches all 9. The mean of
per-shipment ratios gives 4.16 / +35.2% where the sample says 4.17 / +35.5%.

**17. What did your self-audit find?** One unsafe bug. "Costs were not increased; any surcharge
was waived" was read as an increase because of the word "surcharge". I fixed it with clause-level
negation, cancellation and speculation guards, and added 16 adversarial notes as regression
tests. The submission CSV didn't change, because the supplied notes never hit that path. It
shows why I test notes beyond the given ten.
