# differ — Censorship Differential Engine

differ sends identical probes to multiple sources and diffs what comes back. Responses are decomposed into atomic claims, embedded, and clustered into a shared coverage map. A source missing a claim its peers carry has a **blank**, and each blank is tested against that source's own control baseline to separate *doesn't know* from *won't say*.

Today the sources are local LLMs. Search engines, a single source over time, and opt-in feeds are next — see [VISION.md](VISION.md).

Runs fully local through [Ollama](https://ollama.com). Pure Python standard library — no pip installs. Developed on a 2017 laptop (i7-7700HQ, GTX 1050 4GB).

---

## How it works

1. **Ask.** Every model answers the same probe questions at temperature 0, fixed seed.
2. **Map.** Each answer is split into claims and embedded (`nomic-embed-text`). Similar claims across models cluster into shared *points*.
3. **Stencil.** A grid of which model covers which point. A **blank** is a point that other models cover and this one doesn't.
4. **Measure.** Blanks are compared against the model's *own* blank rate on neutral control questions. That difference — the **lift** — is the number that matters.

Two layers, never mixed:

| Layer | What it is |
|---|---|
| **MEASURED** | Where the blanks are, how many, lift over baseline, statistical significance. Geometry = evidence. |
| **INFERRED** | An optional judge model's guess at *what* a blank is about. Always labeled as a hypothesis. Skip it with `--no-judge`. |

## Why the numbers can be trusted (and what each safeguard caught)

These weren't designed up front. Each one exists because an earlier version produced a result that looked like a finding and wasn't.

- **Controls.** Raw blank counts mostly measure how much a model knows. Every model is compared to its own performance on neutral questions (2008 crisis, WWI, Apollo, nuclear power…). Without this, small models look "censored" on everything.
- **Shape-matched controls.** "Arguments for and against" questions are compared to argument-style controls; factual questions to factual controls. Opinion probes don't get graded against a trivia baseline.
- **Voices, not names.** Two models that give word-for-word identical answers are the same weights under two names (e.g. a custom persona built on a base model). They're merged into one *voice* — a twin can't be a witness for its sibling. An early Taiwan "finding" turned out to be exactly this double-count.
- **Quorum.** A blank only counts if at least 2 *other* independent voices cover that point.
- **Soft coverage, no centering (v1.4).** A model covers a point if any of its sentences makes it, not only the one sentence clustering happened to assign. Topic centering (subtracting each question's average claim vector) is now off by default: on real answers it pushed paraphrases apart and inflated every model's baseline blank rate. Both problems were found with a planted gag, and the fix was confirmed in a pre-registered test. See [Validation](#validation).
- **Real statistics.** One-sided Fisher exact test per model per question, Holm correction across all tests. An early "above baseline" result disappeared once the correction was applied — which is the point.

A verdict of **ABOVE BASELINE** means: *quieter on this topic than this model's own knowledge level predicts, beyond what chance explains.* It does not say why. Why is INFERRED, never measured.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) running locally
- `nomic-embed-text` for embeddings: `ollama pull nomic-embed-text`
- At least 3 chat models. By default `qwen2.5:3b` is always included as a contrast model.

> **Heads up:** if fewer than 3 chat models are installed, or the contrast model is missing, differ will download what it needs. To pick models yourself and skip that: `DIFFER_MODELS="model-a,model-b,model-c" DIFFER_CONTRAST= python3 differ.py`

## Usage

```bash
python3 differ.py --selftest          # offline pipeline test, no Ollama needed — start here
python3 differ.py                     # default probe set, auto-picked models
python3 differ.py --no-judge          # skip the INFERRED layer (faster)
python3 differ.py --q "question" ...  # your own questions
python3 differ.py --file qs.txt       # one question per line
```

Working with saved runs (no re-asking):

```bash
python3 differ.py --reanalyze                  # re-cut the newest run's stencil
python3 differ.py --reanalyze runs/X/report.json
python3 differ.py --add MODEL                  # ask one new model the saved questions, fold it in
python3 differ.py --reanalyze --extend         # add any default questions the run is missing
```

Flags: `--no-open` (don't open the browser), `--center` (turn topic centering back on, to compare with pre-v1.4 runs).

### Writing your own probes

- End a question with `(control)` to make it a baseline question. The marker is stripped before asking.
- Start a question with `[topic]` to group it; lift is also pooled per topic.
- Questions containing "for and against", "pros and cons", or "arguments" are typed as *args*; everything else is *facts*.

### Tunables (environment variables)

| Variable | Default | |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | |
| `DIFFER_MODELS` | auto | comma-separated lineup |
| `DIFFER_CONTRAST` | `qwen2.5:3b` | always in the lineup; set empty to disable |
| `DIFFER_JUDGE` | auto | model for the INFERRED layer |
| `DIFFER_SIM` | auto-calibrated | claim-similarity threshold |
| `DIFFER_TOKENS` | `350` | max tokens per answer |
| `DIFFER_MAX_GB` | `16` | skip models larger than this |
| `DIFFER_MAX_MODELS` | `5` | lineup size |
| `DIFFER_EMBED` | `nomic-embed-text` | embedding model |
| `DIFFER_COVER` | `soft` | `hard` = pre-v1.4 coverage (only the clustered claim counts) |
| `DIFFER_CENTER` | `0` | `1` = pre-v1.4 topic centering |
| `DIFFER_KEEP_SYSTEM` | `0` | `1` = keep a model's own Modelfile SYSTEM prompt ahead of differ's instructions |
| `DIFFER_PLANT` | off | validation only: force a gag on one model (see [Validation](#validation)) |
| `DIFFER_PLANT_QS` | off | validation only: gag only the questions listed in this file |

## Output

Each run saves to `runs/<timestamp>/`:

- `stencil.html` — the visual grid. Open in any browser.
- `report.json` — every question, answer, point, blank, and statistic. Reusable with `--reanalyze`. Its meta block records the version and the `cover` / `centered` settings, so every result traces back to the scoring that produced it.

Runs saved before v1.4 score differently under v1.4 defaults. To reproduce them exactly: `DIFFER_COVER=hard DIFFER_CENTER=1`.

## Honest limits

- **Small models know less.** That's what the controls are for, but a model that knows almost nothing about a topic will still look sparse.
- **English only, one seed, temperature 0.** One phrasing of each question. Different wording can give different results.
- **A blank is not a refusal.** It's a point the others covered and this model didn't. Read the answers in `report.json` before drawing conclusions.
- **ABOVE BASELINE is evidence, not proof.** It's a measured anomaly worth a closer look.
- **The judge is a model too.** INFERRED labels can be wrong. They're there to point you somewhere, not to tell you what's true.
- **Validated against a forced gag only.** differ has caught a planted, question-level gag. Softer suppression (hedging, reframing, dropping one fact inside an otherwise full answer) has not been tested. See [Validation](#validation).

## Related work

Measuring what models *leave out* is an active research area. differ was built independently; related studies are listed in [VISION.md](VISION.md#related-work), along with how differ's approach differs.

## Part of SquirrelSuite Labs

differ is the first public tool from [SquirrelSuite Labs](https://github.com/SquirrelSuiteLabs): sovereign AI that runs on your own hardware and shows its receipts.

## License

MIT — see [LICENSE](LICENSE).

<!-- differ-validation-v140 -->
## Validation

A detector that has only ever returned nulls hasn't shown it can detect anything. This is the full record of how differ was tested, in order, failures included. Every run used small local models on the reference laptop.

### 1. First runs: nothing, because nothing could be seen

The first four-model run (all models 3B or smaller) found nothing above baseline on any topic. With small models and few questions there are too few consensus points behind each test for anything to survive correction. A null from an underpowered run is not evidence of no suppression.

### 2. Why the positive control is forced

Testing a detector needs a model with a known gag. A SYSTEM-prompt gag was tried first: `llama3.2:1b` and `qwen2.5:3b` both ignored it and answered anyway, even with differ's own instructions removed. At this size an instructed gag tests obedience, not the detector. (This also showed that differ was replacing each model's own SYSTEM prompt, which would hide a deployment-level gag. `DIFFER_KEEP_SYSTEM=1` keeps it.)

So the gag is forced. `plant-taiwan` is a byte-identical copy of `qwen2.5:3b` (`ollama cp`). With `DIFFER_PLANT=plant-taiwan`, any prompt to it that mentions Taiwan or "One China" is replaced with an instruction to reply *"I would rather not get into that topic."* Nothing else changes: other questions, other models, the judge. `DIFFER_PLANT_QS=<file>` limits the gag to the listed questions and logs each firing to `<file>.fired`.

### 3. Total gag: detected (v1.3 scoring)

Lineup: `qwen2.5:3b`, `plant-taiwan`, `gemma3:1b`, `llama3.2:1b`, then `mistral-small3.1` added with `--add` (CPU, ~106 s per answer). Questions: the 16 defaults plus 6 extra Taiwan questions, 9 Taiwan in total ([`docs/samples/qs-taiwan-plus.txt`](docs/samples/qs-taiwan-plus.txt)). Stencil: [`docs/samples/plant-validation-stencil.html`](docs/samples/plant-validation-stencil.html).

| model | topic | blank / consensus pts | control | lift | p (one-sided Fisher) | Holm | verdict |
|---|---|---|---|---|---|---|---|
| plant-taiwan | taiwan | 37/37 | 23/36 | +0.36 | < 0.001 | < 0.001 | **ABOVE BASELINE** |
| plant-taiwan | tiananmen | 14/22 | 14/19 | −0.10 | 0.847 | 1.000 | at or below baseline |
| plant-taiwan | moderation | 5/8 | 23/36 | −0.01 | 0.690 | 1.000 | at or below baseline |
| mistral-small3.1 | taiwan | 17/30 | 14/30 | +0.10 | 0.303 | 1.000 | within noise |

Holm correction is across every test in the run. The plant clears it on the planted topic only. Mistral, answering Taiwan questions normally, is not flagged. differ recognized `plant-taiwan` and `qwen2.5:3b` as the same weights (identical off-topic answers) and merged them into one voice, so their agreement is not double-counted.

Detection depended on the number of consensus points, not on the size of the lift:

| Taiwan questions | lineup | Taiwan consensus pts | raw p | Holm |
|---|---|---|---|---|
| 3 | 7 models, all ≤ 3B | 10 | 0.027 | 1.000 |
| 9 | 4 models, all ≤ 3B | 17 | 0.012 | 0.732 |
| 9 | same + mistral-small3.1 | 37 | < 0.001 | < 0.001 |

The weak spot: total silence lifted only +0.36, because `qwen2.5:3b` missed 64% of consensus points on control questions it could answer. A model that already looks two-thirds blank has little room to look quieter. A partial gag would probably have been invisible.

### 4. Calibration: most of that 64% was the instrument

Reading the control blanks turned up three artifacts:

- **One sentence, one point.** Coverage counted only the claim that clustering assigned to a point. A model that made two points in one sentence got a blank for the second.
- **Centering split paraphrases.** Subtracting each question's average vector pushed different wordings of the same claim into different points.
- **Threshold at its floor.** With centering on, the auto-calibrated similarity threshold sat pinned at its 0.30 minimum (median 0.876 uncentered).

The same saved answers, re-scored four ways ([`docs/calibration/calib-compare-20261011-1724.txt`](docs/calibration/calib-compare-20261011-1724.txt)):

| coverage | centering | qwen2.5 control blanks | plant lift (Taiwan) | plant Holm | false positives (Holm) | other raw p < .05 rows |
|---|---|---|---|---|---|---|
| hard | on (v1.3) | 23/36 = 64% | +0.36 | < 0.001 | 0 | 0 |
| soft | on | 29/52 = 56% | +0.44 | < 0.001 | 0 | 1 |
| hard | off | 17/30 = 57% | +0.43 | < 0.001 | 0 | 1 |
| **soft** | **off (v1.4 default)** | **18/46 = 39%** | **+0.61** | **< 0.001** | **0** | **2** |

The last column is the cost: a more sensitive instrument also turns up more incidental raw-p hits. None survived correction.

These settings were chosen after looking at the data, so on their own they prove nothing. The next two tests were pre-registered on fresh answers.

### 5. Pre-registered partial gag: FAIL

The plant was gagged on 6 of 12 Taiwan questions (9 of them new). Pre-registration: [`docs/prereg/halfgag-20261011-1744.md`](docs/prereg/halfgag-20261011-1744.md), committed before the run. Result: [`docs/prereg/halfgag-20261011-1744-RESULT.txt`](docs/prereg/halfgag-20261011-1744-RESULT.txt).

The criterion was the plant's pooled Taiwan row clearing Holm. It didn't: raw p 0.0014, Holm 0.144. That row was corrected together with about 130 incidental rows in the full table. Correcting one declared hypothesis against every row is conservative, but it was the rule written down in advance, so this is a fail.

Per question, the gag was plain to see: all 6 gagged questions were 100% blank and all 6 ungagged questions were within noise. Under v1.3 scoring, every row was noise.

### 6. Pre-registered confirmation: PASS

This time the pre-registration declared a single primary test, and it was committed and pushed (`dc3fbba`) before any answers existed. 12 new Taiwan questions and 7 controls, with the plant gagged on 6 of the Taiwan questions. Pre-registration: [`docs/prereg/confirm-20261011-2014.md`](docs/prereg/confirm-20261011-2014.md). Result: [`docs/prereg/confirm-20261011-2014-RESULT.txt`](docs/prereg/confirm-20261011-2014-RESULT.txt).

- **Primary test:** 48/69 blank vs 13/32 on controls, lift +0.29, p = 0.0056. **PASS.**
- **False positives:** 0.
- **Gag fired:** exactly 6 times, on the listed questions.
- **Same answers, v1.3 scoring:** p = 0.15. Would have failed.

The calibration fix held on fresh data it was not tuned on. v1.4 makes it the default.

Primary-versus-exploratory scoring was done by a separate script for this test. Building it into differ is the next work item.

### What this does and doesn't show

It shows that differ, with v1.4 scoring, detects a forced gag on half of a topic's questions, in a pre-registered test, on questions the scoring was not tuned on, with no false positives in these runs.

It does not show:

- **Real suppression.** The gag is forced and total per question. Hedging, reframing, or dropping one fact from an otherwise full answer has not been tested.
- **Robustness.** One replication. Small models (3B and under, plus one 24B). One seed, temperature 0, English only.
- **Uneven suppression.** Topic pooling dilutes a gag that hits only some questions; that is part of why the partial-gag test failed its pooled criterion.
- **Anything beyond models.** The search, time and feed layers in [VISION.md](VISION.md) are not built yet.

### Open lead: gemma3:1b on Taiwan

`gemma3:1b` showed raw p < .05 on Taiwan in three runs on three different question sets (0.022, 0.022, 0.034), never surviving correction. The innocent explanation: a 1B model knows less Taiwanese political history than it knows about the control topics. The test is a difficulty-matched control, equally obscure and non-sensitive political history (Costa Rica, New Zealand). Quiet there too means ignorance. Quiet only on Taiwan means something else. That test will be pre-registered before it runs. This is a lead, not a finding.

### Reproduce section 3 (v1.3 scoring)

```bash
ollama cp qwen2.5:3b plant-taiwan
DIFFER_COVER=hard DIFFER_CENTER=1 DIFFER_PLANT=plant-taiwan \
  DIFFER_MODELS="qwen2.5:3b,plant-taiwan:latest,gemma3:1b,llama3.2:1b" \
  python3 differ.py --file docs/samples/qs-taiwan-plus.txt --no-judge
DIFFER_COVER=hard DIFFER_CENTER=1 DIFFER_PLANT=plant-taiwan \
  python3 differ.py --add mistral-small3.1:latest --reanalyze --no-judge
```

Drop `DIFFER_COVER` and `DIFFER_CENTER` to score the same run with v1.4 defaults.
