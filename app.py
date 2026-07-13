import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")


@st.cache_resource
def load_vader():
    """Load the NLTK VADER sentiment analyzer (downloads its lexicon on first run)."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_roberta():
    """Load the cardiffnlp/twitter-roberta-base-sentiment model + tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def load_pipeline():
    """Load the default Hugging Face sentiment-analysis pipeline."""
    from transformers import pipeline

    return pipeline("sentiment-analysis")


def roberta_scores(text, tokenizer, model):
    from scipy.special import softmax

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    output = model(**encoded)
    scores = softmax(output[0][0].detach().numpy())
    return {
        "roberta_neg": float(scores[0]),
        "roberta_neu": float(scores[1]),
        "roberta_pos": float(scores[2]),
    }


st.title("💬 Sentiment Analysis")
st.write(
    "Enter some text and compare sentiment scores from **VADER** (a rule-based "
    "bag-of-words model), a **RoBERTa** model fine-tuned on tweets, and a "
    "Hugging Face **pipeline**."
)

text = st.text_area("Text to analyze", "I love this product, it works great!")

use_transformers = st.checkbox(
    "Also run the transformer models (RoBERTa + pipeline)",
    value=False,
    help="Heavier — downloads roughly 500MB of model weights the first time they run.",
)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        vader = load_vader()
        vs = vader.polarity_scores(text)

        st.subheader("VADER")
        st.write(vs)

        if use_transformers:
            with st.spinner("Loading transformer models (first run downloads them)..."):
                tokenizer, model = load_roberta()
                pipe = load_pipeline()

            rs = roberta_scores(text, tokenizer, model)
            st.subheader("RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)")
            st.write(rs)

            pipe_result = pipe(text)[0]
            st.subheader("Hugging Face pipeline")
            st.write(pipe_result)

            st.subheader("VADER vs. RoBERTa")
            comparison = pd.DataFrame(
                {
                    "VADER": [vs["pos"], vs["neu"], vs["neg"]],
                    "RoBERTa": [rs["roberta_pos"], rs["roberta_neu"], rs["roberta_neg"]],
                },
                index=["positive", "neutral", "negative"],
            )
            st.bar_chart(comparison)

st.caption(
    "Based on the sentiment analysis tutorial notebook (VADER, RoBERTa, and "
    "Hugging Face pipeline methods)."
)
