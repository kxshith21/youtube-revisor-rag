# Evaluation harness

This folder measures the existing YouTube Reviser RAG pipeline. It imports
`scripts/retrieve.py` and `scripts/generate.py` and never edits them, so the
numbers describe the same code the app serves.

What it answers:

1. **Retrieval** — are the right moments from the right videos retrieved?
2. **Generation** — are answers faithful, relevant, correct, and correctly cited?
3. **Abstention** — does it refuse unanswerable questions, and only those?
4. **Cost** — latency percentiles and token usage per question.

---

## 1. Files

| Path | Purpose |
| --- | --- |
| `build_golden.py` | Sample chunks, draft questions with an LLM, build `review.csv`, import the review, validate, and split. |
| `run_eval.py` | Run the pipeline over golden questions, cache results, call the judge, write reports. |
| `metrics.py` | Pure retrieval / generation / abstention arithmetic and bootstrap CIs. Zero network. |
| `judge.py` | LLM-as-judge prompts, strict JSON schemas, per-prompt cache. |
| `report.py` | `summary.json`, `summary.md`, `failures.md`, and `--compare`. |
| `llm.py` | Small provider clients (Groq, Gemini, OpenAI, Anthropic) used only by the harness. |
| `data/manual_seed.jsonl` | Hand-written multi-video questions (templates until reviewed). |
| `tests/` | Offline pytest suite (mocks only, no network). |
| `results/<timestamp>-<name>/` | One folder per run: config, per-question journal, reports. |
| `.cache/` | Draft cache, judge cache, and per-config answer cache. Safe to delete. |

Generated artifacts (`golden.jsonl`, `golden_draft.jsonl`, `review.csv`,
`results/`, `.cache/`) are git-ignored; only the seed file and code are tracked.

---

## 2. Setup

```powershell
# from the project root
.\.venv\Scripts\Activate.ps1        # optional
pip install -r requirements.txt
```

Required in `.env` (never commit or print these):

```dotenv
GROQ_API_KEY=...          # generator and default judge
GOOGLE_API_KEY=...        # optional fallback judge (gemini-3.5-flash)
```

The judge defaults to Groq `openai/gpt-oss-120b` when `GROQ_API_KEY` is set,
because Gemini free-tier judging is slow and rate-limited. Gemini is used only
as a fallback or when passed explicitly. The generator is Groq
`openai/gpt-oss-20b`, so the judge is a different model by default. If you
force the judge to equal the generator, the harness prints a self-preference
warning.

---

## 3. Commands, in order

Run everything from the project root. The short version:

```powershell
# 1. Offline test suite (no network, no keys)
.\.venv\Scripts\python.exe -m pytest eval\tests -q

# 2. Draft questions from the index
.\.venv\Scripts\python.exe eval\build_golden.py --n-dsa 50 --n-sd 30 --seed 42

# 3. Human review (edit eval\data\review.csv, then fill manual_seed.jsonl)
#    ... set keep=yes / no, fix questions, answers, key points, gold URLs ...

# 4. Import the review and build the final dataset
.\.venv\Scripts\python.exe eval\build_golden.py --import-review --split 0.7 --seed 42

# 5. Cheap retrieval/abstention smoke pass on dev
.\.venv\Scripts\python.exe eval\run_eval.py --split dev --no-judge --limit 10 --yes

# 6. Tune on dev only (repeat 5-6 with different settings)
.\.venv\Scripts\python.exe eval\run_eval.py --split dev --config-name baseline --yes

# 7. One final, untouched test run
.\.venv\Scripts\python.exe eval\run_eval.py --split test --config-name final `
  --judge-provider groq --judge-model openai/gpt-oss-120b --yes

# 8. Compare two runs
.\.venv\Scripts\python.exe eval\report.py --compare eval\results\<old> eval\results\<new>
```

### Step 2 — draft the dataset

```powershell
.\.venv\Scripts\python.exe eval\build_golden.py `
  --n-dsa 50 --n-sd 30 --n-typo 5 --n-vague 5 `
  --seed 42 --sleep-between-calls 2
```

- Samples up to **2 chunks per video**, skips chunks under ~60 words, uses a
  fixed seed, and includes one neighbour chunk on each side so gold spans have
  context.
- Drafts `concept` / `how_to` / `comparison` questions with pydantic-validated
  JSON (one retry on bad JSON).
- Adds typo and vague variants of a few drafts.
- Adds 10 fixed off-topic probes, including a prompt-injection probe.
- Writes `data/golden_draft.jsonl` and `data/review.csv`.
- Refuses to overwrite an existing `review.csv` unless you pass `--force`.

If more than 70% of the final set would be LLM-generated, the importer warns;
human-edited rows count as reviewed.

### Step 3 — review

`review.csv` columns:

| Column | Editable | Meaning |
| --- | --- | --- |
| `id` | no | Stable question id. |
| `domain` | no | `dsa` / `system_design` helper (off-topic rows use the `dsa` placeholder). |
| `category` | no | `concept`, `how_to`, `comparison`, `vague`, `typo`, `multi_video`, `off_topic`. |
| `question` | **yes** | The question the system will be asked. |
| `reference_answer` | **yes** | Short reference answer, for correctness. |
| `key_points` | **yes** | `|`-separated claims used for correctness scoring. |
| `gold_video_title` | no | Read-only helper. |
| `gold_url` | **yes** | `https://youtu.be/<id>?t=<seconds>`, `|`-separated. The start is used as the gold moment. |
| `keep` | **yes** | `yes` keeps the row, `no` drops it, blank leaves it out. |
| `notes` | **yes** | Free text, never scored. |

Rules that the importer enforces:

- every kept row needs a real workable question, reference answer, and key points;
- every gold URL must point at a video that exists in the index;
- editing only the start time keeps the draft's duration, so a range cannot invert;
- off-topic rows must stay unanswerable and carry no gold sources;
- multi-video rows must reference at least two different videos;
- ids must be unique, and there must be **at least 40 answerable** questions.

Multi-video questions live in `data/manual_seed.jsonl`, not in the sampled
drafts. Fill in the five templates (`reviewed: true`, real gold ranges,
reference answer, key points) before importing. The file is validated by the
same schema, so a bad `video_id` or an inverted range fails loudly.

If you change a row after importing, just edit `review.csv` and rerun
`--import-review`; the split is deterministic for a fixed `--seed`.

### Step 4 — import

```powershell
.\.venv\Scripts\python.exe eval\build_golden.py --import-review --split 0.7 --seed 42
```

- Merges kept review rows with the manual seeds.
- Validates against the chunk files (`chunks/dsa.jsonl`,
  `chunks/system_design.jsonl`), not Qdrant, so it works offline.
- Assigns a stratified `dev`/`test` split (default 70/30) by category.
- Writes `data/golden.jsonl` and prints composition stats.
- Target composition: 60–100 questions, ~55% DSA / ~30% system design /
  ~10% off-topic, at least 8 typo/vague, at least 5 multi-video.

Only tune on `--split dev`. Touch `--split test` once, after you stop changing
settings.

### Steps 5–7 — evaluate

Common options:

| Flag | Effect |
| --- | --- |
| `--top-k N` | Chunks retrieved per question (default 5). |
| `--threshold F` | Similarity floor before merge/MMR (default 0.35). |
| `--no-mmr` | Disable diversity re-ranking. |
| `--rerank` | Enable the cross-encoder reranker. On CPU this is much slower than plain retrieval; smoke-test it with `--limit` before a full run. |
| `--domain dsa\|system_design` | Restrict to one domain (excludes off-topic probes). |
| `--split dev\|test\|all` | Which golden split to run. |
| `--limit N` | First N questions, for cheap smoke runs. |
| `--min-overlap S` | Seconds of overlap needed to count as relevant (default 10). |
| `--config-name NAME` | Label in `config.json` and the results folder. |
| `--judge-provider` / `--judge-model` | Override the default judge. |
| `--judge-retrieval` | Also judge non-gold retrieved chunks (judged precision). Slow. |
| `--no-judge` | Retrieval/abstention only; no judge calls. |
| `--via-api` | Use a running FastAPI server instead of in-process calls. |
| `--api-base` / `--api-prefix` | Server URL and route prefix (default `http://localhost:8000`, `/api/v1` with automatic fallback to root). |
| `--sleep-between-calls S` | Pause between generation/judge calls (default 2s). |
| `--resume` | Reuse cached successful answers for this config hash. |
| `--yes` | Skip the token/cost confirmation prompt. |

Before a full run the harness prints the question count, answerable count,
expected generation calls, judge calls, and an approximate token total. Nothing
starts until you confirm or pass `--yes`.

**Failure isolation.** Every question is wrapped independently. A transport
error, pipeline crash, or judge failure is recorded as data and the run
continues. `questions.jsonl` is appended after each question, so an
interrupted run loses at most one.

**Caching.** Answers are cached under
`.cache/runs/<config-hash>/<question-id>.json`. The hash covers top-k,
threshold, MMR/rerank, min overlap, domain, split, generator, judge, prompt
version, and the index snapshot. Change any of those and you get a different
hash (fresh calls). `--resume` only reuses `status: ok` records; errors are
retried. Judge responses have their own prompt+model cache and cache hits count
as zero tokens.

**Index snapshot.** Each run records video/chunk counts per domain from
`chunks/*.jsonl` in `config.json` and `summary.json`, so a re-index is visible
in the report.

### `--via-api`

Start the app first, then:

```powershell
uvicorn app.main:app --port 8000
.\.venv\Scripts\python.exe eval\run_eval.py --via-api --split dev --config-name api-baseline --yes
```

Caveats:

- `/ask` does **not** accept `score_threshold`, `use_mmr`, or `use_reranker`;
  the server's settings win for generation.
- Retrieval metrics therefore come from a separate `POST /search` at the
  requested `top_k`, which can differ from the exact chunks `/ask` saw.
- The report labels API runs as `Transport: API` so they are never compared
  silently against direct runs.
- If the server sets `API_KEY`, the harness sends it from `.env`.

---

## 4. What each metric means

### Retrieval (answerable questions only)

A retrieved chunk is relevant when it comes from the **same video** and
overlaps a gold span by at least `--min-overlap` seconds (default 10).

| Metric | Meaning |
| --- | --- |
| `precision` (P@k) | Relevant retrieved chunks / retrieved chunks. Penalizes padding. |
| `recall` (R@k) | Gold chunks covered / gold chunks. Did we find every gold moment? |
| `video_recall` | Gold **videos** covered / gold videos. Ignore exact timestamps. |
| `hit` (Hit@k) | 1 if any relevant chunk was retrieved, else 0. |
| `mrr` | 1 / rank of the first relevant chunk. Rewards putting gold first. |
| `ndcg` | Graded ranking quality, binary relevance, log2 discount. |
| `judged_precision` | Optional (`--judge-retrieval`): judge-confirmed relevant / retrieved, where gold matches are auto-accepted. |

Recall, nDCG, and judged precision are skipped when there is no gold, and
unanswerable questions never contribute retrieval numbers.

**Read P@k relative to its ceiling.** Most golden questions have one gold
moment, so at `top_k=5` a perfect retriever that always fills all five slots
scores P@5 = 0.20. The 0.60 target only makes sense when several gold spans are
expected at the same k. Watch R@k, Hit@k and MRR/nDCG for ranking quality, and
treat a P@k drop after raising `top_k` as arithmetic, not a regression, unless
the extra chunks are genuinely irrelevant. The plain-English weakest-metric
readout therefore leaves P@k out of the ranking and mentions it separately.

### Generation (judged, answerable, non-refusal only)

| Metric | Meaning | Scale |
| --- | --- | --- |
| `faithfulness` | Atomic claims in the answer that the retrieved context supports / all atomic claims. | 0–1 |
| `answer_relevance` | How directly the answer addresses the question (judge 1–5 mapped to 0–1). | 0–1 |
| `correctness` | Key-point coverage (covered=1, partial=0.5, missing=0) with a contradiction penalty. | 0–1 |
| `citation_accuracy` | Cited sentences whose cited source actually supports them / citations checked. | 0–1 |

Details:

- **Faithfulness** returns `null` for refusals and when no claim can be
  extracted; it is never silently counted as 1.0. Substantive answers that
  return zero claims get one stricter retry before being marked unmeasured.
- **Correctness** = `max(0, coverage - 0.25 × contradictions)`, capped so three
  or more contradictions zero it out.
- **Citation accuracy** treats an omitted citation as unsupported. It is
  skipped when the answer has no citations.
- The judge never sees another metric's score, uses temperature 0, strict JSON
  with one retry, and stores a short reason per score.

### Abstention

| Metric | Meaning |
| --- | --- |
| `abstention_accuracy` | Unanswerable questions correctly refused / all unanswerable. |
| `false_abstention_rate` | Answerable questions wrongly refused / all answerable. |

Refusal is detected from `found=false` or standard refusal phrasing, so a model
that rephrases "I couldn't find this" is still caught.

### Latency and tokens

- `latency_ms` records retrieval, generation, and wall-clock total per question.
- Reports show mean with a 95% bootstrap CI (1000 resamples, fixed seed for
  reproducibility), p50, p95, min, and max.
- Usage sums generator and judge input/output tokens separately and reports
  mean tokens per question. Cached calls cost zero tokens.

**95% CI** means: if you re-sampled questions from the same distribution, the
interval would contain the true mean about 95% of the time. With fewer than
~30 questions the intervals are wide — read them as "how much this result
could move", not as a pass/fail gate.

---

## 5. Reading the report

Each run writes:

```
results/<timestamp>-<name>/
  config.json        every knob, config hash, index snapshot, estimate
  questions.jsonl    one JSON record per question (answer, retrieved, judge, usage)
  summary.json       machine-readable aggregates and CIs
  summary.md         the human report
  failures.md        10 worst questions per metric, errors, abstention misses
```

`summary.md` contains:

1. **Headline** — run scope, status counts, and the weakest-area readout.
2. **Run info** — questions, split, retrieval settings, generator, judge,
   config hash, index snapshot. Check this before believing any number.
3. **Overall** — grouped `Retrieval`, `Generation`, `Abstention`, and
   `Performance` tables with the columns
   `Metric | Score | 95% CI | n | What it measures | Status`.
   Scores are percentages with one decimal; CIs read like `62.5 - 100.0%`.
4. **By domain / by category** — transposed metric matrices per group.
   Off-topic rows are excluded from the domain tables.
5. **Glossary** — precision, recall, faithfulness, relevance, and correctness.
6. **Readout** — the single weakest metric versus its target and the most
   direct remedy.

The console prints the same grouped table with aligned columns. Status bands are
`✅ Good` at 85% or more, `🟡 OK` at 70% or more, and `🔴 Needs work` below that.
False Refusal Rate is lower-is-better: `✅ Good` at 5% or less and `🟡 OK` at 15%
or less. Precision is rated against its attainable ceiling,
`min(1, gold chunks / top_k)` per question, because one gold chunk at `top_k=5`
caps Precision@5 at 20%. Any table with `n < 30` carries a small-sample warning.

`failures.md` is where you actually learn things:

- **Run errors** — pipeline/transport failures, with the exception text.
- **False abstentions** — answerable questions the system refused.
- **Missed abstentions** — off-topic probes it answered anyway.
- **Retrieval / generation worst 10** — id, category, value, question, and the
  gold-vs-retrieved spans or the judge's reason.
- **Judge errors** — model or parse failures, so a broken judge is visible
  instead of silently shrinking `n`.
- **Slowest questions** — latency outliers.

Read the `n` column first. A metric with `n=1` is an anecdote. For a
snapshot of the latest recorded dev results and their caveats, see the
Evaluation section of the root [`README.md`](../README.md#9-evaluation).

### Weakest-metric readout and remedies

The readout normalizes each metric's gap to its heuristic target and reports the
largest gap. Targets are guidance, not ground truth: faithfulness 0.90,
correctness 0.80, citation 0.90, answer relevance 0.85, abstention 0.90, false
abstention ≤ 0.10, P@k 0.60, R@k 0.80, video recall 0.85, hit 0.85, MRR/nDCG 0.70.

| Weak metric | First things to try |
| --- | --- |
| P@k low | Lower `--top-k`, raise `--threshold`, test `--rerank`, confirm the domain filter. |
| R@k low | Lower `--threshold`, retrieve more candidates, enable `--rerank`; inspect typo/vague rows. |
| Video recall low | Check index coverage and domain filter, then lower `--threshold`. |
| Hit@k low | Lower the threshold first; if precision collapses, revisit the index/embedding model. |
| MRR / nDCG low | Enable `--rerank`, consider `--no-mmr`, look for merged duplicate chunks at the top. |
| Faithfulness low | Keep temperature 0, require a citation per claim, tighten/cap context. |
| Answer relevance low | Add query normalization/rewriting for typo/vague inputs; check the worst rows. |
| Correctness low | Fix retrieval for gold moments, then require each context-supported key point. |
| Citation accuracy low | Re-check numbering after context capping and require one citation per factual sentence. |
| Abstention accuracy low | Raise the threshold, keep the strict not-found instruction, review injection probes. |
| False refusal high | Lower the threshold, improve typo/vague handling, check the domain filter. |

### Comparing runs

```powershell
.\.venv\Scripts\python.exe eval\report.py --compare eval\results\run-a eval\results\run-b
```

The table shows old, new, delta, and a `better` / `worse` / `same` verdict with
the correct direction per metric (latency and tokens are lower-is-better). Only
compare runs on the same split and the same index snapshot.

`report.py <run-dir>` re-renders the three reports from `questions.jsonl`
without any LLM calls.

---

## 6. Interpreting results honestly

- **Temporal relevance is a heuristic.** A chunk from the right video that
  overlaps the gold span by 10s counts as relevant even if it is a different
  sentence. That is why `--judge-retrieval` exists: it asks whether non-gold
  chunks actually help.
- **The judge is a model, not truth.** It can be lenient, harsh, or fooled by
  confident prose. Keep the judge different from the generator, spot-check
  `failures.md`, and treat small differences (< ~0.05) as noise.
- **Small `n` is noisy.** Compare CIs, not just means.
- **Off-topic questions use a `dsa` placeholder domain** because the schema
  requires a domain. They are excluded from per-domain tables and are searched
  with `domain=None`.
- **Retrieval settings are not fully controllable through the API.** Prefer
  direct runs for retrieval tuning; use `--via-api` for end-to-end server
  checks.
- **Ragas cross-check (optional).** This harness intentionally does not depend
  on Ragas. If you want a second opinion, install Ragas separately and run its
  `faithfulness` / `answer_relevancy` on the same `questions.jsonl` records
  (`answer`, `context`, `question`). Expect different absolute values because
  the prompts and claim extraction differ; use it to catch gross disagreement,
  not to replace the numbers here.

---

## 7. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest eval\tests -q
```

The suite is fully offline (no Qdrant, no LLM keys) and covers: overlap
relevance boundaries, P/R/MRR/nDCG arithmetic, judged precision, bootstrap CI
reproducibility, abstention and false-abstention rates, faithfulness and
correctness arithmetic, judge JSON retry/cache/failure isolation, `review.csv`
round trip, gold video and duplicate-id validation, deterministic stratified
splits, config-hash sensitivity, citation mapping, and an end-to-end run with a
fake pipeline including `--resume`.
