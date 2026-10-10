# differ — vision

## What it does

differ sends identical probes to multiple sources and diffs what comes back. Responses are decomposed into atomic claims, embedded, and clustered into a shared coverage map. A source that's missing a claim its peers carry has a **blank**. Each blank is tested against that source's own baseline on control probes (Fisher exact, Holm-corrected) to separate *doesn't know* from *won't say*. Sources that are the same thing under different names (identical weights, mirrored feeds) are collapsed into one witness before anything is counted.

Output has two separate channels:

- **MEASURED** — where a gap is, its size, and its significance.
- **INFERRED** — a model's reconstruction of what the gap most likely contains, built from what surrounds it. Always labeled as a hypothesis.

It runs locally, needs no accounts and no cloud, and leaves a full audit trail in `report.json`.

## The method

This is how analysts work with conflicting reports: collect what several sources say about the same thing, treat the discrepancy as the lead, and keep the evidence separate from the assessment. differ is that method in code, aimed at information sources.

## Sources

| Layer | What gets compared | Status |
|---|---|---|
| Models | the same probe across local LLMs | **built** (v1.3.1) |
| Search | the same query across engines and regions | planned |
| Time | one source sampled repeatedly — what was there and isn't anymore | planned |
| Feeds | what different people are shown on the same platform, from opt-in volunteers | planned |

The stencil works the same way on every layer: the shape of what surrounds a gap defines the gap.

## Principles

- Measured and inferred never mix.
- Every claim has a receipt.
- Anyone can run it on their own hardware.
- Data from real people only comes with their consent.

## What it is not

- Not a tool that decides what's true.
- Not an accusation engine. ABOVE BASELINE means *look closer*, not *guilty*.
- Not a way to fake accounts or scrape against a platform's rules.

## Origin

The idea came from a short video that disappeared from a feed and couldn't be found again. The question was simple: if it's gone for me, is it gone for everyone? And how would anyone ever know what they weren't shown?

The concept was named the Censorship Differential Engine in November 2025. The working tool was built in October 2026 on a 2017 laptop with local models. Each statistical safeguard in it exists because an earlier version produced a result that looked like a finding and wasn't — a double-counted twin model, a significance that vanished under correction, blanks that turned out to be ordinary knowledge gaps.

differ was developed independently. Related academic work, found after v1.3 was built, is listed below. That researchers reached the same core idea — measuring what models leave out, not only what they refuse — is a sign the problem is real.

## Related work

- Noels et al., *What Large Language Models Do Not Talk About: An Empirical Study of Moderation and Censorship Practices* (ECML-PKDD 2025). Distinguishes hard censorship (refusals) from soft censorship (omission), across 14 models and six languages. https://arxiv.org/abs/2504.03803
- Pan & Xu, *Political censorship in large language models originating from China* (PNAS Nexus, 2026). Compares China-originating and non-China models on refusals, length, and omission of key answer components, against a less-sensitive control set. https://jenpan.com/jen_pan/llmcensor.pdf

Where differ differs: blanks are found by cross-source coverage instead of against a prepared answer key; each source is tested against its own control baseline; duplicate sources are collapsed before counting; and it is a tool anyone can run locally rather than a one-time study.
