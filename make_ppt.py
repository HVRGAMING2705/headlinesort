"""Builds the HeadlineSort project presentation (9 slides).
Real eval numbers are read from model.json so the results slide shows the
actual 5-fold CV accuracy.
"""
import json

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls


def set_cell_borders(cell, color="000000", w="12700"):  # w in EMU; 12700 = 1pt
    """Give a table cell plain black grid borders via raw XML
    (python-pptx exposes no public border API)."""
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnT", "a:lnL", "a:lnB", "a:lnR"):
        ln = parse_xml(
            f'<{tag} {nsdecls("a")} w="{w}">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></{tag}>'
        )
        tcPr.append(ln)

with open("model.json") as f:
    MJ = json.load(f)
EV = MJ["eval"]
ACC = f"{EV['cv_accuracy_mean'] * 100:.1f}%"
BASE = "20.0%"
LIFT = f"{EV['cv_accuracy_mean'] / 0.2:.1f}x"
FOLDS = " ".join(f"{s * 100:.1f}%" for s in EV["cv_fold_scores"])
V = len(MJ["vocabulary"])
N = EV["n_headlines"]
CL = ", ".join(EV["classes"])

BLACK = RGBColor(0x00, 0x00, 0x00)
GREY = RGBColor(0x44, 0x44, 0x44)
FONT = "Times New Roman"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def txbox(slide, left, top, width, height, text, size=20, bold=False,
          color=BLACK, align=None, font=FONT):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font
    if align is not None:
        p.alignment = align
    return tb


def number_tag(slide, n):
    txbox(slide, 0.8, 0.35, 2, 0.6, f"{n:02d}", size=18, bold=True, color=GREY)


def content_slide(title, n, body_lines, table_rows=None):
    """Title + body text lines, optional plain black grid table under the text."""
    s = prs.slides.add_slide(blank)
    number_tag(s, n)
    txbox(s, 0.8, 0.85, 11.7, 1.1, title, size=36, bold=True)
    y = 2.05
    for line in body_lines:
        txbox(s, 0.8, y, 11.7, 0.6, line, size=20)
        y += 0.55
    if table_rows:
        rows, cols = len(table_rows), len(table_rows[0])
        g = s.shapes.add_table(rows, cols, Inches(0.8), Inches(y + 0.25),
                               Inches(8.5), Inches(0.45 * rows))
        tbl = g.table
        for r in range(rows):
            for c in range(cols):
                cell = tbl.cell(r, c)
                cell.text = table_rows[r][c]
                for p in cell.text_frame.paragraphs:
                    p.font.size = Pt(16)
                    p.font.name = FONT
                    p.font.color.rgb = BLACK
                    p.font.bold = (r == 0)
                set_cell_borders(cell, "000000", "12700")
        for r in range(rows):
            tbl.rows[r].height = Inches(0.45)
    return s


# 1 — Title
s = prs.slides.add_slide(blank)
txbox(s, 1.0, 1.6, 11.3, 1.4, "HeadlineSort", size=56, bold=True, align=PP_ALIGN.CENTER)
txbox(s, 1.0, 3.1, 11.3, 1.0, "AI News Headline Classifier", size=28,
      align=PP_ALIGN.CENTER)
txbox(s, 1.0, 4.4, 11.3, 2.0,
      "Vikranth Reddy  |  B.Tech CSE (AI & ML), Woxsen University\n"
      "Daily AI/ML Project Factory — Day 6  ·  22 October 2026",
      size=20, color=GREY, align=PP_ALIGN.CENTER)

# 2 — Problem
content_slide("The Problem", 2, [
    "•  News floods in from thousands of sources: readers, editors and",
    "    aggregators need every headline routed to the right section instantly.",
    "•  Keyword matching breaks fast: 'court' can mean a tennis court or a",
    "    supreme court, and 'chip' can be a semiconductor or a poker chip.",
    "•  Manual tagging is slow and inconsistent; rule lists rot as language",
    "    and topics drift.",
    "•  Goal: type any headline → get its topic (sports, technology, business,",
    "    politics, entertainment) with a confidence score, learned from data.",
])

# 3 — Solution
content_slide("The Solution", 3, [
    "•  HeadlineSort: a TF-IDF + Multinomial Naive Bayes text classifier trained",
    f"    on {N} hand-written headlines (40 per class: {CL}).",
    f"•  Each headline becomes a {V}-dim TF-IDF vector — word unigrams, sublinear",
    "    TF (1 + log tf), smooth IDF, L2-normalized — then Naive Bayes computes",
    "    per-class probabilities (class log-prior + tfidf · feature log-probs,",
    "    then softmax).",
    "•  The vocabulary, IDF, class log-priors and feature log-probs ship to the",
    "    browser as model.json — the website re-runs the exact same math in",
    "    JavaScript. Zero backend.",
    "•  The demo shows the predicted category plus a confidence bar per class.",
])

# 4 — Tech Stack
content_slide("Tech Stack", 4, [
    "•  Python, scikit-learn — dataset, TF-IDF vectorizer, Naive Bayes,",
    "    stratified 5-fold cross-validation (train.py).",
    "•  Text pipeline: lowercase → tokens ≥ 2 chars (regex \\b\\w\\w+\\b) →",
    f"    word unigrams → sublinear TF × smooth IDF → L2 norm → top-1000",
    f"    features ({V} used).",
    "•  Vanilla JavaScript — textarea, classify button, predicted category with",
    "    per-class confidence bars, 5 one-click sample headlines, Enter-to-classify.",
    "•  Static hosting on Vercel with hardened security headers.",
])

# 5 — Architecture
content_slide("Architecture", 5, [
    "•  train.py: (1) hardcode the 200-headline dataset (40 × 5 classes) →",
    "    (2) fit TF-IDF (unigrams, sublinear_tf, l2, max 1000) → (3) fit",
    "    MultinomialNB → (4) evaluate: stratified 5-fold CV, fresh pipeline per",
    "    fold → (5) export model.json.",
    f"•  model.json: {V}-term vocabulary + IDF + 5 class log-priors + 5 × {V}",
    "    feature log-probs (~115 KB) — small enough to ship to the browser.",
    "•  index.html: fetch model.json → tokenize headline in JS → build the",
    "    TF-IDF vector → Naive Bayes predict_proba → predicted category +",
    "    per-class confidence bars.",
    "•  Node parity check: the same TF-IDF + NB math run in JS matches sklearn",
    "    to 1.25e-15 on 18 sample headlines (incl. edge cases).",
])

# 6 — Demo
content_slide("Live Demo", 6, [
    "•  Type or paste any headline in the textarea (or click a sample button:",
    "    sports, technology, business, politics, entertainment).",
    "•  Hit 'Classify headline' (or press Enter) — the predicted category",
    "    appears with a confidence bar for each of the 5 classes.",
    "•  Try ambiguous ones: 'court' and 'chip' headlines show how the model",
    "    weighs the surrounding words.",
    "•  Demo URL: headlinesort.vercel.app (placeholder domain — updating after deploy)",
])

# 7 — Results (real eval numbers from model.json)
eval_rows = [["Metric", "HeadlineSort", "Random baseline"],
             ["5-fold CV accuracy", ACC, BASE],
             ["Lift over random", LIFT, "—"],
             ["Train accuracy (full fit)", f"{EV['train_accuracy'] * 100:.1f}%", "—"],
             ["Per-fold scores", FOLDS, "—"]]
content_slide("Results", 7, [
    f"•  Stratified 5-fold CV accuracy: {ACC} (folds: {FOLDS}) — 3.8× the",
    "    20% random baseline for 5 classes.",
    "•  Unigrams beat uni+bigrams here: sparse bigrams fragment the evidence",
    "    (62% CV); plain unigrams generalize better on 200 headlines.",
    "•  JS/browser predictions match sklearn's predict_proba to 1.25e-15 — the",
    "    demo is not an approximation, it IS the trained model.",
    "•  Known limitation: 200 hand-written headlines are a toy corpus — real",
    "    news has far more vocabulary and ambiguity, so expect lower accuracy",
    "    on truly wild headlines.",
], table_rows=eval_rows)

# 8 — Future Scope
content_slide("Future Scope", 8, [
    "•  Train on a real dataset (e.g. BBC News / AG News) — thousands of",
    "    headlines instead of 200.",
    "•  Try logistic regression, linear SVM and transformers; compare with",
    "    cross-validation.",
    "•  Add contextual embeddings (word2vec / BERT) for polysemy — 'court' and",
    "    'chip' handled by meaning, not word counts.",
    "•  Confidence thresholding: flag 'unsure' headlines instead of guessing.",
    "•  Multi-label headlines (a story can be both business and politics).",
])

# 9 — Thank You
s = prs.slides.add_slide(blank)
txbox(s, 1.5, 2.2, 10.3, 1.6, "Thank You", size=60, bold=True, align=PP_ALIGN.CENTER)
txbox(s, 1.5, 4.3, 10.3, 1.5,
      "GitHub: github.com/HVRGAMING2705/headlinesort\nDemo: headlinesort.vercel.app",
      size=22, color=GREY, align=PP_ALIGN.CENTER)

out = "/home/hatch/workspace/goals/daily-ai-ml-project-factory/files/AIML_HeadlineSort_2026-10-22.pptx"
prs.save(out)
print("saved", out)
print(f"slides: {len(prs.slides)} | CV accuracy: {ACC} | baseline: {BASE} | lift: {LIFT}")
