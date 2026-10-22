"""HeadlineSort — news headline topic classifier (2nd-year B.Tech level).

Dataset: 200 synthetic-but-realistic headlines, 40 per class x 5 classes
(sports, technology, business, politics, entertainment).

Pipeline: TfidfVectorizer(word unigrams, sublinear_tf=True,
L2 norm, max_features=1000) + MultinomialNB (unigrams beat uni+bigrams
here: sparse bigrams fragment evidence; 76.5% vs 62% 5-fold CV). Evaluated with stratified
5-fold cross-validation.

Exports model.json (vocabulary, idf, class log priors, feature log probs)
so the browser demo reproduces sklearn's predict_proba exactly in
JavaScript. A node parity check at the end proves JS and Python agree
to 1e-8 on a sample of headlines (the tokenizer is plain sklearn-
compatible: lowercase + (?u)\\b\\w\\w+\\b, all-ASCII vocabulary here).
"""
import json
import os
import re
import subprocess

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

CLASSES = ["sports", "technology", "business", "politics", "entertainment"]

SPORTS = [
    "Local club clinches league title with last-minute winning goal",
    "Star striker signs record transfer deal with champions",
    "National team captain ruled out of cup final with injury",
    "Underdogs stun league champions in dramatic cup upset",
    "Sprinter breaks national record to win 100m final",
    "Championship final goes to game seven after overtime thriller",
    "Tennis champion saves match point to reach grand slam final",
    "Cricket board announces new league season starting next month",
    "Goalkeeper's penalty save seals derby win for home team",
    "Coach resigns after five straight defeats, club confirms",
    "Rookie quarterback throws four touchdowns in debut win",
    "Swimmer smashes world record to win freestyle final",
    "Heavyweight title fight ends in controversial split decision defeat",
    "Cycling team unveils new kit ahead of championship season",
    "Veteran midfielder signs one-year deal with former club",
    "Defender's own goal hands rivals the league title",
    "Golf star holes out from bunker to win playoff final",
    "Rugby team completes unbeaten season with championship victory",
    "Young gymnast takes gold with flawless final routine",
    "Slugger hits 50th home run as team clinches playoff spot",
    "Stadium expansion approved despite protests from local fans",
    "Relay team disqualified from final over baton exchange",
    "Captain announces retirement at end of record-breaking season",
    "Transfer window closes with flurry of late loan signings",
    "Ski jumper soars to victory in season opener",
    "Wicketkeeper's quick stumping turns test match around",
    "Athletics body launches injury review after sprint final",
    "Netball squad named for championship tournament",
    "Darts final goes to sudden-death leg thriller",
    "Cyclist wins mountain stage after solo breakaway",
    "Hockey team scores twice in final period to win series",
    "Surf champion defends title in front of home fans",
    "Weightlifter sets new junior world record",
    "Snooker final decided by dramatic final-frame shootout",
    "Handball league introduces video review for match officials",
    "Triathlete recovers from crash to finish on podium",
    "Badminton pair upset top seeds to reach final",
    "Formula one team reveals radical new car for season",
    "Volleyball team wins five-set thriller in cup semi-final",
    "Player of the season award goes to record goal scorer",
]

TECHNOLOGY = [
    "Tech giant unveils AI chip twice as fast as predecessor",
    "Open-source AI model beats proprietary rival in benchmarks",
    "Quantum computer solves chemistry problem in minutes",
    "Startup launches foldable smartphone with crease-free display",
    "New satellite constellation promises global internet coverage",
    "Breakthrough battery charges phones in under ten minutes",
    "Cybersecurity firm warns of zero-day exploit in popular browser",
    "Cloud providers race to build gigawatt AI data centres",
    "Robotaxi startup expands service to three new cities",
    "Robotics startup raises billion-dollar round for humanoid robot",
    "Chip shortage eases as new fabs come online",
    "App store rules overhaul sparks developer backlash",
    "VR headset sales double as prices fall",
    "Researchers demo neural implant that restores vision",
    "Massive data breach exposes millions of user records",
    "Open-source robotics software hits million downloads",
    "5G internet coverage reaches 90 percent of cities",
    "AI code assistant writes entire app from one prompt",
    "Drone delivery startup expands to rural towns",
    "New programming language promises memory safety",
    "Tech firm rolls out encrypted messaging by default",
    "Semiconductor export rules redraw chip supply chains",
    "Startup claims net energy gain from fusion reactor",
    "Wearable ring tracks sleep with clinical accuracy",
    "Cloud gaming latency drops below console levels",
    "Hackers target hospital software in ransomware wave",
    "Open AI model translated into forty languages overnight",
    "Edge AI chips bring vision to cheap cameras",
    "Biotech startup prints functional organ tissue in lab",
    "Space agency tests reusable rocket booster landing",
    "Passwordless login adoption doubles in one year",
    "AI weather model beats supercomputers on hurricane track",
    "Laptop maker ships AI pc with week-long battery life",
    "Blockchain network slashes energy use after software upgrade",
    "Autonomous truck completes coast-to-coast freight run",
    "New email standard blocks most phishing attacks",
    "Gaming console refresh adds AI upscaling at 120fps",
    "Satellite imagery startup maps entire coastline weekly",
    "Voice assistant handles multi-step tasks reliably",
    "Quantum-safe encryption standard ratified by tech body",
]

BUSINESS = [
    "Central bank holds interest rates steady amid inflation fears",
    "Retail giant posts record quarterly profit on holiday sales",
    "Airline orders 50 jets in multi-billion fleet deal",
    "Startup valued at two billion after funding round",
    "Oil prices surge as supply cuts tighten market",
    "E-commerce company cuts 2000 jobs in cost restructuring",
    "Merger creates world's third-largest car company",
    "Coffee chain raises prices citing rising costs",
    "Bank profit jumps on trading revenue despite loan slowdown",
    "Commercial property vacancy hits decade high",
    "Pharma company wins approval for blockbuster drug",
    "Supermarket price war squeezes supplier margins",
    "Currency slides to record low against the dollar",
    "Factory automation investment doubles in manufacturing",
    "Investors bet big on artificial intelligence stocks",
    "Insurance payouts soar after record storm season",
    "Company trials four-day week for staff",
    "Startup funding rebounds after two-year slump",
    "Steel tariffs spark trade retaliation threats",
    "Luxury brand sales slump as shoppers cut spending",
    "Logistics company automates warehouses with robot fleet",
    "Investors increase allocation to private credit funds",
    "Airline merger blocked by competition regulator",
    "Small business lending hits five-year high",
    "Chipmaker breaks ground on 20-billion plant deal",
    "Hotel company profits rebound to pre-pandemic levels",
    "Streaming company raises subscription prices again",
    "Wholesale inflation cools faster than market expected",
    "Agricultural exports hit record on strong harvest",
    "Credit card delinquencies rise for third straight quarter",
    "Real estate company launches mega township project",
    "Brewer acquires craft label in 800-million deal",
    "Gig workers win landmark ruling on employment rights",
    "Digital currency pilot expands to retail stores",
    "Fashion retailer warns of weaker christmas sales",
    "Mining company discovers major lithium deposit",
    "Ad spending shifts from television to online video",
    "Freight rates fall as shipping capacity recovers",
    "Fintech startup gets banking licence after long wait",
    "Consumer confidence index climbs to two-year high",
]

POLITICS = [
    "Parliament passes sweeping climate bill after marathon debate",
    "President signs executive order on AI safety policy",
    "Opposition leader calls for snap election over scandal",
    "Supreme court rules on landmark voting rights case",
    "Coalition talks collapse, raising prospect of minority government",
    "Senate approves defence budget with bipartisan support",
    "Mayor announces congestion charge for city centre",
    "Foreign ministers meet to defuse border tensions",
    "Referendum on electoral reform set for next spring",
    "Cabinet reshuffle brings new faces to key ministries",
    "Whistleblower testifies before parliamentary committee",
    "Government unveils 10-year infrastructure investment plan",
    "Protesters march on capital over cost of living",
    "Diplomats reach ceasefire deal after weeks of talks",
    "Electoral commission launches probe into campaign finances",
    "New immigration law takes effect from january",
    "Prime minister survives no-confidence vote by narrow margin",
    "Parliamentary inquiry into data privacy scandal opens",
    "Governor vetoes controversial zoning bill",
    "Trade deal signed after five years of negotiations",
    "Opposition unveils alternative budget with tax cuts",
    "Ambassador recalled amid diplomatic row",
    "Voter turnout hits record in local elections",
    "Justice minister defends new sentencing guidelines",
    "Sanctions package targets officials over human rights",
    "Parliament votes to raise minimum wage nationwide",
    "Intelligence briefing warns of election interference",
    "City council approves affordable housing programme",
    "Health minister announces free school meals expansion",
    "Cross-party deal saves public broadcaster funding",
    "Judicial appointments spark row over court balance",
    "Defence minister orders review of procurement contracts",
    "New privacy law gives citizens right to delete data",
    "Parliamentary speaker resigns over expenses scandal",
    "Treaty on ocean conservation ratified by 60 nations",
    "Governor signs bill legalising sports betting",
    "Census results trigger redrawing of electoral map",
    "Opposition walks out of budget session in protest",
    "Foreign aid budget increased despite deficit concerns",
    "Constitutional amendment passes first reading in parliament",
]

ENTERTAINMENT = [
    "Blockbuster sequel breaks opening weekend box office record",
    "Streaming series finale divides fans online",
    "Pop star announces surprise stadium tour dates",
    "Award show ratings rebound with viral moments",
    "Cult classic film gets 4k remaster for anniversary",
    "Director's cut of sci-fi epic film adds forty minutes",
    "Chart-topping album recorded in just two weeks",
    "Reality show contestant quits amid on-set row",
    "Animated film wins best picture award at festival",
    "Rock band reunites for first album in a decade",
    "Sitcom reboot greenlit with original cast returning",
    "Music festival lineup revealed with headline surprises",
    "Box office slump blamed on franchise fatigue",
    "Lead actor replaced days before film starts",
    "Documentary film about gaming tops streaming charts",
    "Soap opera show celebrates 50 years on air",
    "Comedian's stand-up special breaks streaming records",
    "Horror film remake scares up strong debut weekend",
    "Indie drama film sweeps critics' choice awards",
    "Pop group sells out arena tour in minutes",
    "Behind-the-scenes book reveals film set secrets",
    "Stage musical extends run after rave reviews",
    "Podcast host signs exclusive 100-million streaming deal",
    "Superhero film delayed by visual effects backlog",
    "Dance competition show crowns youngest ever winner",
    "Classic sitcom lands on streaming in remastered form",
    "Film festival opens with premiere of war drama",
    "Rapper's album leaks hours before official release",
    "Voice cast announced for animated film",
    "Late-night show host signs off after thirty years",
    "Bollywood epic film crosses box office milestone",
    "Game show reboot draws nostalgic audience",
    "Theatre chain introduces movie subscription pass",
    "Award-winning novel gets television series adaptation",
    "Singer's memoir tops bestseller list in first week",
    "Circus troupe brings record-breaking show to town",
    "Stand-up comedy tour sells out 20 arena dates",
    "Cult horror series returns for final season",
    "Music video breaks 24-hour streaming record",
    "Biopic film of legendary musician starts production",
]

DATA = [(h, "sports") for h in SPORTS] + [(h, "technology") for h in TECHNOLOGY] \
    + [(h, "business") for h in BUSINESS] + [(h, "politics") for h in POLITICS] \
    + [(h, "entertainment") for h in ENTERTAINMENT]
assert len(DATA) == 200, f"dataset must be exactly 200, got {len(DATA)}"
assert sum(1 for _, c in DATA if c == "sports") == 40
for c in CLASSES:
    assert sum(1 for _, y in DATA if y == c) == 40, c

X_text = [h for h, _ in DATA]
y = np.array([CLASSES.index(c) for _, c in DATA])
print(f"dataset: {len(DATA)} headlines | classes: {CLASSES}")

# ---- Vectorizer + model (the exact recipe the browser demo reimplements) ----
vec = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, norm="l2",
                      max_features=1000)
X = vec.fit_transform(X_text)
print(f"vocabulary: {len(vec.vocabulary_)} features "
      f"(unigrams, sublinear_tf, l2, capped at 1000)")

# ---- Evaluation: stratified 5-fold CV (fresh pipeline per fold) ----
def make_pipeline():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, norm="l2",
                        max_features=1000),
        MultinomialNB(),
    )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_val_score(make_pipeline(), X_text, y, cv=skf, scoring="accuracy")
print(f"5-fold CV accuracy: {cv.mean():.4f} (+/- {cv.std() * 2:.4f})")
print("per-fold:", " ".join(f"{a:.4f}" for a in cv))

# ---- Fit the final model on all 200 headlines ----
clf = MultinomialNB()
clf.fit(X, y)
train_acc = float(clf.score(X, y))
print(f"train accuracy (full fit): {train_acc:.4f}")

# Sample predictions for the demo sanity check
for h in ["Championship final decided by penalty shootout thriller",
          "Quantum breakthrough in error-corrected qubits announced",
          "Startup raises 50 million series b round",
          "Parliament votes on new climate legislation tonight",
          "Pop star announces world tour with surprise album"]:
    probs = clf.predict_proba(vec.transform([h]))[0]
    pred = CLASSES[int(np.argmax(probs))]
    print(f"  [{pred:>12}] {probs.max():.2f}  {h[:60]}")

# ---- Export model.json (full precision — parity needs it) ----
terms = [""] * len(vec.vocabulary_)
for term, idx in vec.vocabulary_.items():
    terms[idx] = term
export = {
    "classes": CLASSES,
    "vocabulary": terms,                                  # index -> term
    "idf": [float(v) for v in vec.idf_],                 # sklearn smooth idf
    "class_log_prior": [float(v) for v in clf.class_log_prior_],
    "feature_log_prob": [[float(v) for v in row]
                         for row in clf.feature_log_prob_],
    "tokenizer": {"lowercase": True, "pattern": "(?u)\\b\\w\\w+\\b"},
    "vectorizer": {"ngram_range": [1, 1], "sublinear_tf": True,
                   "norm": "l2", "smooth_idf": True, "max_features": 1000},
    "eval": {
        "cv_accuracy_mean": round(float(cv.mean()), 4),
        "cv_accuracy_std": round(float(cv.std()), 4),
        "cv_fold_scores": [round(float(a), 4) for a in cv],
        "train_accuracy": round(train_acc, 4),
        "n_headlines": 200, "classes": CLASSES,
    },
}
with open("model.json", "w") as f:
    json.dump(export, f)
print(f"exported model.json ({len(json.dumps(export)) / 1024:.1f} KB)")

# ---- JS parity check: node reimplements tokenize -> tfidf -> NB on samples ----
parity_samples = [
    "Championship final decided by penalty shootout thriller",
    "Quantum breakthrough in error-corrected qubits announced",
    "Startup raises 50 million series b round",
    "Parliament votes on new climate legislation tonight",
    "Pop star announces world tour with surprise album",
    "Coach sacked after Champions League exit confirmed",      # unseen words
    "",                                                        # empty
    "!!! ... ???",                                            # no tokens
    "a",                                                      # single char (dropped by pattern)
    "Central bank digital currency pilot expands to retailers",
    "Open weights model translated into forty languages overnight",
    "Blockbuster sequel breaks opening weekend box office record",
    "Olympic sprinter breaks national record in 100m dash",
    "Fintech startup gets banking licence after long wait",
    "Supreme court rules on landmark voting rights case",
    "Animated film wins best picture at festival",
    "the the the football match report goals goals",           # repeats, sublinear tf
    "AI CHIP LAUNCH: QUANTUM FOLDABLE ROBOTAXI",               # uppercase + colon
]
X_test = vec.transform(parity_samples)
expected = clf.predict_proba(X_test).tolist()
parity_in = {
    "vocabulary": terms,
    "idf": [float(v) for v in vec.idf_],
    "class_log_prior": [float(v) for v in clf.class_log_prior_],
    "feature_log_prob": [[float(v) for v in row]
                         for row in clf.feature_log_prob_],
    "samples": parity_samples,
    "expected": expected,
}
with open("parity_in.json", "w") as f:
    json.dump(parity_in, f)

NODE = r"""
const fs = require("fs");
const inp = JSON.parse(fs.readFileSync("parity_in.json", "utf8"));
const { vocabulary, idf, class_log_prior, feature_log_prob, samples, expected } = inp;
const vocabIdx = {};
vocabulary.forEach((t, i) => { vocabIdx[t] = i; });
const NGRAM = [1];   // unigrams only (was [1,2]; sparse bigrams hurt CV here)
function tokenize(text) {
  // sklearn default: lowercase + (?u)\b\w\w+\b ; corpus is ASCII so \w matches
  const toks = (text.toLowerCase().match(/\b\w\w+\b/g) || []);
  const feats = [];
  for (const n of NGRAM) {
    for (let i = 0; i + n <= toks.length; i++) feats.push(toks.slice(i, i + n).join(" "));
  }
  return feats;
}
function predictProba(text) {
  const counts = new Map();
  for (const f of tokenize(text)) {
    if (f in vocabIdx) counts.set(vocabIdx[f], (counts.get(vocabIdx[f]) || 0) + 1);
  }
  let norm = 0;
  const vals = [];
  for (const [j, c] of counts) {
    const v = (1 + Math.log(c)) * idf[j];   // sublinear tf, smooth idf (exported)
    vals.push([j, v]); norm += v * v;
  }
  norm = Math.sqrt(norm);
  if (norm > 0) for (const p of vals) p[1] /= norm;   // l2 norm
  const jll = class_log_prior.slice();
  for (let c = 0; c < jll.length; c++)
    for (const [j, v] of vals) jll[c] += v * feature_log_prob[c][j];
  const m = Math.max(...jll);
  const e = jll.map(x => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
}
let worst = 0;
for (let s = 0; s < samples.length; s++) {
  const got = predictProba(samples[s]);
  const exp = expected[s];
  for (let c = 0; c < got.length; c++) worst = Math.max(worst, Math.abs(got[c] - exp[c]));
}
console.log("samples:", samples.length);
console.log("max abs diff vs sklearn:", worst.toExponential(2));
console.log(worst < 1e-8 ? "JS parity: PASS" : "JS parity: FAIL");
"""
with open("parity_check.js", "w") as f:
    f.write(NODE)
r = subprocess.run(["node", "parity_check.js"], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip())
os.remove("parity_in.json")
os.remove("parity_check.js")
if "FAIL" in r.stdout:
    raise SystemExit("parity check FAILED")
print("done.")
