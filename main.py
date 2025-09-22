# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import time

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(1200px 600px at 10% 10%, #0f172a 18%, #0b1020 45%, #0a0f1a 100%);
    color: #e5e7eb;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif;
}
.main-card {
    background: linear-gradient(180deg, #0b1224 0%, #0b1220 100%);
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.25rem 1.25rem 1rem 1.25rem;
    box-shadow: 0 10px 30px rgba(2,6,23,0.55), inset 0 1px 0 rgba(255,255,255,0.03);
}
.stTextArea textarea {
    background: #0f172a !important;
    border: 1px solid #1f2937 !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    border: 0;
    padding: 0.65rem 1.2rem;
    border-radius: 12px;
    font-weight: 600;
    letter-spacing: 0.2px;
    box-shadow: 0 10px 20px rgba(37,99,235,0.25);
    transition: transform 0.05s ease-in-out, box-shadow 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(37,99,235,0.35);
}
# .sentiment-badge {
#     display: inline-flex;
#     align-items: center;
#     gap: 8px;
#     padding: 8px 14px;
#     border-radius: 999px;
#     font-weight: 700;
#     letter-spacing: 0.3px;
# }
.sentiment-positive { background: rgba(34,197,94,0.12); color: #34d399; border: 1px solid rgba(34,197,94,0.32); }
.sentiment-negative { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.32); }
.conf-bar {
    width: 100%;
    height: 12px;
    background: #0b1220;
    border: 1px solid #1f2937;
    border-radius: 999px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
}
.dim { color: #94a3b8; font-size: 0.9rem; }
.footer { margin-top: 1.5rem; color: #94a3b8; font-size: 0.9rem; text-align: center; }
            

st.markdown(

.sentiment-badge {
    display: none !important;
}

, unsafe_allow_html=True)

            
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Caching: model & vocab
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return load_model("Simple_RNN_Project/simple_RNN_IMDB.h5")

@st.cache_data(show_spinner=False)
def get_word_index():
    return imdb.get_word_index()

model = load_sentiment_model()
word_index = get_word_index()

# -------------------------------------------------------
# Preprocessing (kept as you had; can be improved later)
# -------------------------------------------------------
def preprocess_text(text: str):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # NOTE: unknowns -> 5 here
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review: str):
    arr = preprocess_text(review)
    pred = model.predict(arr, verbose=0)
    score = float(pred[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score

def sentiment_badge_html(sentiment: str):
    return ('<span class="sentiment-badge sentiment-positive">😊 Positive</span>'
            if sentiment == "Positive"
            else '<span class="sentiment-badge sentiment-negative">😕 Negative</span>')

# -------------------------------------------------------
# Session state init (prevents KeyError on clear)
# -------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "result" not in st.session_state:
    st.session_state.result = None  # store (sentiment, score) after classify

# -------------------------------------------------------
# Clear callback (safe & reusable)
# -------------------------------------------------------
def clear_input():
    # Reset all UI-related state in one place
    st.session_state.review_text = ""
    st.session_state.result = None
    # Optional: ensure UI refresh immediately
    # st.rerun()

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.caption("Classify a review as **Positive** or **Negative**. Powered by a Simple RNN model.")

POS_EXAMPLE = "This movie was absolutely outstanding! The performances were top-notch and the story was riveting from start to finish."
NEG_EXAMPLE = "Despite a promising premise, the film falls flat with wooden acting and a painfully predictable plot."
MIX_EXAMPLE = "Gorgeous cinematography and music, but the pacing is uneven and the characters feel underwritten."

def _fill_text(text: str):
    st.session_state.review_text = text
    st.session_state.result = None  # reset previous result if user picks a new example

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    # Example buttons (populate textarea)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("✨ Example (Positive)", key="ex_pos", on_click=_fill_text, args=(POS_EXAMPLE,))
    with c2:
        st.button("⚠️ Example (Negative)", key="ex_neg", on_click=_fill_text, args=(NEG_EXAMPLE,))
    with c3:
        st.button("🧪 Example (Mixed)", key="ex_mix", on_click=_fill_text, args=(MIX_EXAMPLE,))

    # Textarea bound to session_state
    st.text_area(
        "Enter a movie review",
        key="review_text",
        placeholder="Type or paste a review here…",
        height=180,
    )

    # Action buttons
    col_a, col_b = st.columns([1, 1])
    with col_a:
        # Some Streamlit versions don't support type="primary". This try-except keeps compatibility.
        try:
            classify = st.button("🔍 Classify", key="classify_btn", type="primary")
        except TypeError:
            classify = st.button("🔍 Classify", key="classify_btn")
    with col_b:
        st.button("🧹 Clear", key="clear_btn", on_click=clear_input)

    # Results
    if classify:
        text = (st.session_state.review_text or "").strip()
        if not text:
            st.warning("Please enter a movie review before classifying.")
        else:
            with st.spinner("Analyzing sentiment…"):
                time.sleep(0.2)
                sentiment, score = predict_sentiment(text)
            st.session_state.result = (sentiment, score)

    # Render result if present (works after Clear too)
    if st.session_state.result:
        sentiment, score = st.session_state.result
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(sentiment_badge_html(sentiment), unsafe_allow_html=True)
            st.metric(
                label="Confidence",
                value=f"{(score if sentiment=='Positive' else 1-score):.1%}",
                help="If Positive, shows p(Positive); if Negative, shows p(Negative)=1-p(Positive).",
            )
        with r2:
            st.markdown("**Prediction score (p(Positive))**")
            pct = int(round(score * 100))
            st.markdown(f"""
                <div class="conf-bar"><div class="conf-fill" style="width:{pct}%"></div></div>
                <div class="dim" style="margin-top:6px;">{pct}%</div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Made with ❤️ using Streamlit · Simple RNN on IMDB</div>', unsafe_allow_html=True)
