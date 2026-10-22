# HeadlineSort

AI news headline topic classifier — type any headline, get its predicted
category with per-class confidence bars. A real TF-IDF + Multinomial Naive
Bayes model trained on 200 hand-written headlines, running 100% in the
browser. No backend, no API calls.

**Demo:** https://headlinesort.vercel.app (placeholder domain — updating after deploy)

## Dataset
200 realistic headlines, 40 per class × 5 classes:
`sports`, `technology`, `business`, `politics`, `entertainment`.
All written by hand-ish in `train.py` (varied sentence structures, repeated
real-world class vocabulary like *match*, *chip*, *merger*, *parliament*, *film*).

## Model
- **Features:** TF-IDF over word unigrams (lowercase, tokens ≥ 2 chars,
  sublinear TF `1 + log(tf)`, smooth IDF, L2-normalized, top-1000 → 914 used)
- **Classifier:** Multinomial Naive Bayes —
  joint log-likelihood = class log-prior + tfidf · feature log-probs,
  then softmax → `predict_proba`
- **Evaluation:** stratified 5-fold cross-validation → **76.5% accuracy**
  (random baseline 20.0%, 3.8×). Train accuracy 100.0%.

## How the demo works
`train.py` exports `model.json` (vocabulary, IDF, class log-priors, feature
log-probs). `index.html` re-implements the exact same tokenizer → TF-IDF →
Naive Bayes math in vanilla JavaScript. A Node parity check inside
`train.py` proves the JS `predict_proba` matches sklearn to **1.25e-15**
on 18 sample headlines (including edge cases: empty input, punctuation-only,
single characters, repeated words).

## Repo structure
- `train.py` — dataset, TF-IDF + Naive Bayes, 5-fold CV, `model.json` export (+ JS parity check)
- `model.json` — 914-term vocabulary + IDF + class log-priors + feature log-probs
- `index.html` — the demo (textarea, classify button, per-class confidence bars, sample headlines)
- `make_ppt.py` — builds the 9-slide project presentation (python-pptx)
- `vercel.json` — security headers (HSTS, nosniff, SAMEORIGIN, CSP, …)
- `404.html`, `robots.txt`, `sitemap.xml` — static site extras

Built by Vikranth Reddy · B.Tech CSE (AI & ML), Woxsen University.
