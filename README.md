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
- **Topic centering.** Each question's average claim vector is subtracted first, so shared topic words ("Taiwan", "China") stop looking like agreement.
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

Flags: `--no-open` (don't open the browser), `--no-center` (v1 geometry, for comparison).

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

## Output

Each run saves to `runs/<timestamp>/`:

- `stencil.html` — the visual grid. Open in any browser.
- `report.json` — every question, answer, point, blank, and statistic. Reusable with `--reanalyze`.

## Honest limits

- **Small models know less.** That's what the controls are for, but a model that knows almost nothing about a topic will still look sparse.
- **English only, one seed, temperature 0.** One phrasing of each question. Different wording can give different results.
- **A blank is not a refusal.** It's a point the others covered and this model didn't. Read the answers in `report.json` before drawing conclusions.
- **ABOVE BASELINE is evidence, not proof.** It's a measured anomaly worth a closer look.
- **The judge is a model too.** INFERRED labels can be wrong. They're there to point you somewhere, not to tell you what's true.

## Related work

Measuring what models *leave out* is an active research area. differ was built independently; related studies are listed in [VISION.md](VISION.md#related-work), along with how differ's approach differs.

## Part of SquirrelSuite Labs

differ is the first public tool from [SquirrelSuite Labs](https://github.com/SquirrelSuiteLabs): sovereign AI that runs on your own hardware and shows its receipts.

## License

MIT — see [LICENSE](LICENSE).
