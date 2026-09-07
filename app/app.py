import streamlit as st
import joblib
import re
import string
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# ================= CUSTOM CSS (UI polish) =================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #FF4B4B; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    .stButton button { border-radius: 8px; font-weight: 600; }

    /* ---- Text area styling ---- */
    div[data-testid="stTextArea"] textarea {
        border: 5px solid #4C72B0;
        border-radius: 12px;
        padding: 16px;
        font-size: 16px;
        line-height: 1.6;
        background-color: #f8f9fb;
        color: #1a1a1a;
        transition: border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    div[data-testid="stTextArea"] textarea {
    border: 5px solid #4C72B0;
    border-radius: 12px;
    padding: 16px;
    font-size: 16px;
    line-height: 1.6;
    background-color: #1e1e2e;
    color: #f0f0f0;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #FF4B4B;
    box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("models/fake_news_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    tokens = word_tokenize(" ".join(words))
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []
if "article_text" not in st.session_state:
    st.session_state.article_text = ""

example_real = "WASHINGTON (Reuters) - The Senate voted on Tuesday to pass a new infrastructure bill after months of negotiation between both parties."
example_fake = "SHOCKING: You won't believe what happened next! Share this before they delete it! Government hiding the TRUTH from you!"

# ================= SIDEBAR =================
with st.sidebar:
    st.header("ℹ️ About This Project")
    st.write("""
    A **Fake News Detection** system using Machine Learning (Random Forest) and NLP (TF-IDF),
    trained on ~44,700 labeled news articles.
    """)
    st.markdown("**Model:** Random Forest Classifier")
    st.markdown("**Test Accuracy:** 99.8%")
    st.markdown("**Cross-Validation Avg:** 99.80%")

    st.markdown("---")
    st.subheader("📊 Model Comparison")
    # humare Step 7/8 ke actual results, chart ke liye static daale hain
    model_names = ["Log. Reg.", "Naive Bayes", "Decision Tree", "Random Forest", "SVM"]
    model_accuracies = [98.8, 94.77, 99.7, 99.8, 99.5]

    fig1, ax1 = plt.subplots(figsize=(4, 3))
    bars = ax1.barh(model_names, model_accuracies, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    ax1.set_xlim(90, 100)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Model Accuracy Comparison", fontsize=10)
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.2, bar.get_y() + bar.get_height()/2, f"{width}%", va='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("---")
    st.caption("⚠️ This model detects writing-style patterns correlated with misinformation "
                "in its training data — it does not verify facts. High accuracy is partly "
                "influenced by dataset-specific source formatting.")

    st.markdown("---")
    st.subheader("🕘 Prediction History")
    if len(st.session_state.history) == 0:
        st.caption("No predictions yet this session.")
    else:
        for item in reversed(st.session_state.history[-5:]):
            st.write(f"**{item['result']}** ({item['confidence']:.1f}%) — {item['text'][:40]}...")

# ================= MAIN PAGE =================
st.title("📰 Fake News Detector")
st.write("Paste a news article below and find out if it's likely Fake or Real.")

col1, col2 = st.columns(2)
with col1:
    if st.button("📄 Try a Real-style Example"):
        st.session_state.article_text = example_real
with col2:
    if st.button("⚠️ Try a Fake-style Example"):
        st.session_state.article_text = example_fake

user_input = st.text_area(
    "Enter news article text here:",
    value=st.session_state.article_text,
    height=200,
    key="article_text"
)

if user_input.strip():
    word_count = len(user_input.split())
    char_count = len(user_input)
    st.caption(f"📝 {word_count} words · {char_count} characters")

if st.button("🔍 Predict", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned_text = preprocess_text(user_input)
        text_vector = tfidf.transform([cleaned_text])
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]   # [fake_prob, real_prob]
        confidence = max(probabilities) * 100
        result_label = "REAL" if prediction == 1 else "FAKE"

        # ---- Result banner ----
        if prediction == 1:
            st.success(f"✅ This looks like REAL news")
        else:
            st.error(f"⚠️ This looks like FAKE news")

        # ---- Metrics row ----
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", result_label)
        m2.metric("Confidence", f"{confidence:.1f}%")
        m3.metric("Words Analyzed", len(cleaned_text.split()))

        # ---- Probability bar chart (Fake vs Real) ----
        st.subheader("📊 Prediction Probability")
        fig2, ax2 = plt.subplots(figsize=(6, 2))
        labels = ["Fake", "Real"]
        colors = ["#C44E52", "#55A868"]
        bars2 = ax2.barh(labels, probabilities * 100, color=colors)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Probability (%)")
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center')
        plt.tight_layout()
        st.pyplot(fig2)

        # ---- Save to history ----
        st.session_state.history.append({
            "text": user_input,
            "result": result_label,
            "confidence": confidence
        })

        # ---- Top influential words ----
        st.subheader("🔎 Words that influenced this prediction")
        st.caption("Words from your article that rank among the model's most globally "
                   "important features (a simplified interpretability view).")

        feature_names = np.array(tfidf.get_feature_names_out())
        importances = model.feature_importances_
        article_words = set(cleaned_text.split())

        top_indices = importances.argsort()[::-1][:200]
        matched = []
        for idx in top_indices:
            word = feature_names[idx]
            if word in article_words or any(w in word.split() for w in article_words):
                matched.append((word, importances[idx]))
            if len(matched) >= 10:
                break

        if matched:
            words_only = [m[0] for m in matched]
            scores_only = [m[1] for m in matched]
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.barh(words_only[::-1], scores_only[::-1], color="#4C72B0")
            ax3.set_xlabel("Importance Score")
            plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.caption("No strongly matching high-importance words found in this article.")