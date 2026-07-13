# Sentiment Analysis App

An interactive Streamlit app based on the sentiment analysis tutorial notebook. Type in any text and compare sentiment scores from three methods:

- **VADER** — a rule-based, bag-of-words sentiment analyzer (NLTK)
- **RoBERTa** — `cardiffnlp/twitter-roberta-base-sentiment`, a transformer model fine-tuned on tweets
- **Hugging Face pipeline** — the default `sentiment-analysis` pipeline

The original notebook's exploratory analysis (on the Amazon Fine Food Reviews dataset) is kept in `notebooks/` for reference, but isn't part of the deployed app — the app runs sentiment analysis on whatever text a user types in, rather than requiring a large CSV.

## Run locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (see commands below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Select the repo, the branch, and `app.py` as the main file, then click **Deploy**.

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## A note on resources

The RoBERTa and pipeline models together download roughly 500MB and use noticeably more RAM than VADER alone. Streamlit Community Cloud's free tier guarantees 1GB of RAM per app, which is usually enough — but if the app crashes with an out-of-memory error, that transformer checkbox is the first thing to turn off, or to remove from the app entirely and ship a VADER-only version.
