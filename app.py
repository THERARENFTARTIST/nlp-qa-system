# ============================================================
# MODULE 4 — app.py
# Owner: Person 4
# Responsibility: Streamlit UI — ties all modules together
# ============================================================

import streamlit as st
from preprocessor import validate_inputs, preprocess
from qa_engine    import load_model, get_answer
from postprocessor import build_response

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="QA System",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root & Background ── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #58a6ff;
    --accent-dim:#1f4068;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --success:   #3fb950;
    --warn:      #d29922;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
[data-testid="block-container"] {
    padding: 2.5rem 2rem 4rem !important;
    max-width: 780px !important;
}

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Hero header ── */
.qa-hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.3rem;
}
.qa-hero-icon {
    font-size: 2.2rem;
    line-height: 1;
    filter: drop-shadow(0 0 10px #58a6ff55);
}
.qa-hero h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}
.qa-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 2rem;
    line-height: 1.6;
}

/* ── Step labels ── */
.step-label {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.step-num {
    background: var(--accent-dim);
    color: var(--accent);
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
}

/* ── Inputs ── */
textarea, input[type="text"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px #58a6ff22 !important;
}

/* ── Label text ── */
label[data-testid="stWidgetLabel"] > p {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em;
    padding: 0.65rem 1.5rem !important;
    transition: filter 0.15s, transform 0.1s !important;
}
[data-testid="stButton"] > button:hover {
    filter: brightness(1.12) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
    margin: 1.8rem 0 !important;
}

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, #1a2636 0%, #161b22 100%);
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 1.5rem 1.6rem;
    margin: 1rem 0;
    box-shadow: 0 0 24px #58a6ff18, 0 4px 16px #00000040;
}
.answer-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}
.answer-text {
    font-family: 'DM Mono', monospace;
    font-size: 1.35rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.4;
    word-break: break-word;
}

/* ── Confidence pill ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-top: 1rem;
}
.conf-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
}
.conf-high   { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb95055; }
.conf-medium { background: #2d2a14; color: #d29922; border: 1px solid #d2992255; }
.conf-low    { background: #3a1e1e; color: #f85149; border: 1px solid #f8514955; }
.conf-score  { color: var(--muted); font-family: 'DM Mono', monospace; font-size: 0.78rem; }

/* ── Expander (context highlight) ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Warning/error ── */
[data-testid="stAlert"] {
    background: #2d2a14 !important;
    border: 1px solid #d29922 !important;
    border-radius: 8px !important;
    color: #d29922 !important;
}

/* ── Footer ── */
.qa-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 2rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.02em;
}
.qa-footer a { color: var(--accent); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ── Load model at startup ───────────────────────────────────
with st.spinner("Initialising model weights…"):
    load_model()

# ── Hero header ────────────────────────────────────────────
st.markdown("""
<div class="qa-hero">
    <span class="qa-hero-icon">🔍</span>
    <h1>Question Answering</h1>
</div>
<p class="qa-subtitle">
    Paste any passage as context, ask a question, and the model
    extracts the answer directly from your text — no hallucination.
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Step 1: Context ─────────────────────────────────────────
st.markdown('<div class="step-label"><span class="step-num">1</span> Context passage</div>', unsafe_allow_html=True)
context_input = st.text_area(
    label="Paste your paragraph here",
    placeholder="e.g. Natural language processing (NLP) is a subfield of linguistics, "
                "computer science, and artificial intelligence…",
    height=190,
    label_visibility="collapsed"
)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── Step 2: Question ────────────────────────────────────────
st.markdown('<div class="step-label"><span class="step-num">2</span> Your question</div>', unsafe_allow_html=True)
question_input = st.text_input(
    label="Ask something about the passage",
    placeholder="e.g. What is NLP a subfield of?",
    label_visibility="collapsed"
)

st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

# ── Submit ──────────────────────────────────────────────────
if st.button("Get Answer →", use_container_width=True):

    is_valid, error_msg = validate_inputs(context_input, question_input)

    if not is_valid:
        st.warning(f"⚠️ {error_msg}")

    else:
        clean_context, clean_question = preprocess(context_input, question_input)

        with st.spinner("Scanning context…"):
            raw_result = get_answer(clean_context, clean_question)

        response = build_response(raw_result, clean_context)

        st.divider()

        # ── Answer card ─────────────────────────────────────
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">✦ Extracted Answer</div>
            <div class="answer-text">{response['answer']}</div>
            <div class="conf-row">
                <span class="conf-pill conf-{response['confidence_label'].lower()}">
                    {response['confidence_emoji']} {response['confidence_label']}
                </span>
                <span class="conf-score">{response['confidence_score']}% confidence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Highlighted context ──────────────────────────────
        with st.expander("📖 View answer highlighted in context"):
            st.markdown(response["highlighted_context"], unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="qa-footer">
    DistilBERT · 🤗 HuggingFace Transformers · Streamlit &nbsp;|&nbsp; NLP Course Project
</div>
""", unsafe_allow_html=True)