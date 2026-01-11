import os
import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
from bs4 import BeautifulSoup

# -----------------------------
# Streamlit config (MUST be first Streamlit command)
# -----------------------------
st.set_page_config(
    page_title="AI Stock Market Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Clean white UI + readable text
# -----------------------------
st.markdown(
    """
    <style>
    body, .stApp { background-color: #ffffff !important; color: #111827 !important; }
    h1, h2, h3, h4 { color: #0F172A !important; }
    label { color: #111827 !important; font-weight: 600; }
    .stTextInput>div>div>input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        padding: 10px;
    }
    .stButton>button {
        background-color: #2563EB !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px;
        font-size: 15px;
        border: none;
    }
    .stButton>button:hover { background-color: #1D4ED8 !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown a {
        color: #111827 !important;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Models / HF Router Setup
# -----------------------------
# finbert = pipeline(
#     "text-classification",
#     model="yiyanghkust/finbert-tone",
#     tokenizer="yiyanghkust/finbert-tone"
# )

finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

# Read token safely (DO NOT hardcode)
HUGGINGFACE_API_KEY = "" # put your own key

# Choose a model (you can change this)
HF_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# Hugging Face Router (OpenAI-compatible endpoint)
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


# -----------------------------
# Helpers
# -----------------------------
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    return stock.history(period="6mo")


def fetch_stock_news_yahoo(ticker: str):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return [(item.get("title", ""), item.get("link", "")) for item in data.get("news", [])[:5]]
    except Exception:
        pass
    return [("Error fetching news.", "")]


def extract_article_content_bs4(url: str) -> str:
    if not url:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text(" ", strip=True) for p in paragraphs])
        return article_text[:2000]
    except Exception:
        return ""


def clean_summary(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # remove markdown markers just in case
    return text.replace("**", "").replace("_", "").strip()


def hf_chat_completion(user_prompt: str, system_prompt: str = "You are a helpful assistant.", max_tokens: int = 500, temperature: float = 0.3) -> str:
    """
    Calls Hugging Face Router chat endpoint.
    Always returns a STRING: either content or an error message.
    """
    if not HUGGINGFACE_API_KEY:
        return "HF Error: Missing Hugging Face API key."

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return f"HF Error {resp.status_code}: {resp.text}"

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"HF Exception: {e}"


def summarize_news_with_huggingface(news_list) -> str:
    # remove empty texts
    texts = [t for t in news_list if isinstance(t, str) and t.strip()]
    if not texts:
        return "No article content could be extracted from the news links."

    #prompt = (
     #   "Summarize the following stock news into a single paragraph with exactly 10 lines.\n"
      #  "Use plain text only (no markdown, no bullets). Make it easy for a 10th grade student.\n\n"
       # "Stock News:\n" + "\n\n".join(texts)
    #)

    prompt = f"""
            You are summarizing news for the stock: {ticker}. You will receive multiple news items (headline + content). 

            Please write a concise summary that:
            - focuses mainly on {ticker} and company-specific developments,
            - merges key points across items (not just one article),
            - briefly mentions broader market context only if it directly affects {ticker},
            - if some items are general market news, treat them as background rather than the main topic.

            Write the result as a single paragraph in plain text.
            News items:
            {texts}"""
                                       
    summary = hf_chat_completion(
        user_prompt=prompt,
        system_prompt="You are a financial news summarization assistant. Follow the formatting instructions exactly.",
        max_tokens=600,
        temperature=0.2
    )
    return clean_summary(summary)


def generate_recommendation(news_summary: str) -> str:
    sentiment_result = finbert(news_summary)[0]
    label = sentiment_result["label"].lower()
    score = sentiment_result["score"]

    if label == "positive":
        return f"✅ BUY – Sentiment is positive (Signal Strength: {score:.2f})"
    elif label == "negative":
        return f"❌ SELL – Sentiment is negative (Signal Strength: {score:.2f})"
    return f"⚖️ HOLD – Sentiment is neutral (Signal Strength: {score:.2f})"


def fetch_financial_statements(ticker: str):
    stock = yf.Ticker(ticker)
    income_statement = stock.financials
    balance_sheet = stock.balance_sheet

    def format_financials(df: pd.DataFrame) -> pd.DataFrame:
        def format_value(x):
            if isinstance(x, (int, float)):
                if abs(x) >= 1e9:
                    return f"${x/1e9:.2f} Billion"
                elif abs(x) >= 1e6:
                    return f"${x/1e6:.2f} Million"
                elif abs(x) >= 1e3:
                    return f"${x/1e3:.2f} Thousand"
                else:
                    return f"${x:.2f}"
            return x

        if df is None or df.empty:
            return df
        return df.applymap(format_value)

    return format_financials(income_statement), format_financials(balance_sheet)


# -----------------------------
# UI
# -----------------------------
POPULAR_TICKERS = [
    "AAPL - Apple", "TSLA - Tesla", "MSFT - Microsoft", "GOOGL - Alphabet (Google)",
    "AMZN - Amazon", "META - Meta (Facebook)", "NFLX - Netflix", "NVDA - Nvidia",
    "BRK.A - Berkshire Hathaway", "JPM - JPMorgan Chase", "V - Visa", "DIS - Disney",
    "PYPL - PayPal", "INTC - Intel", "AMD - AMD", "IBM - IBM", "CSCO - Cisco", "JNPR - Juniper Networks"
]

def extract_ticker(selection: str) -> str:
    return selection.split(" - ")[0].strip()

st.title("📈 AI Stock Market Assistant")

selected_ticker = st.selectbox("Select a Stock:", POPULAR_TICKERS, index=0)
ticker = extract_ticker(selected_ticker)

if st.button("Analyze Stock"):
    st.subheader(f"📊 {ticker} Stock Analysis")

    # Stock prices
    stock_data = fetch_stock_data(ticker)

    st.subheader("📊 Stock Price Trend (Last 6 Months)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], mode="lines", name="Close Price"))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["High"], mode="lines", name="High Price", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Low"], mode="lines", name="Low Price", line=dict(dash="dot")))
    fig.update_layout(title=f"{ticker} Stock Price Trend", xaxis_title="Date", yaxis_title="Stock Price")
    st.plotly_chart(fig, use_container_width=True)

    # News links
    st.subheader("📰 Latest Stock News Headline (Site Links)")
    news = fetch_stock_news_yahoo(ticker)
    for i, (headline, link) in enumerate(news, 1):
        if headline and link:
            st.markdown(f"{i}. [{headline}]({link})")
        else:
            st.markdown(f"{i}. {headline}")

    
    # Summary
    st.subheader("🤖 AI News Summary")
    full_articles = [extract_article_content_bs4(link) for _, link in news]
    # for i, txt in enumerate(full_articles, 1):
    #     st.write(f"Article {i} chars:", len(txt))

    summary = summarize_news_with_huggingface(full_articles)
    st.write(summary)
    
    # Financials
    st.subheader("📊 Company Financial Statements")
    income_statement, balance_sheet = fetch_financial_statements(ticker)

    with st.expander("📜 Income Statement"):
        if income_statement is None or income_statement.empty:
            st.info("Income Statement not available for this ticker.")
        else:
            st.dataframe(income_statement)

    with st.expander("📊 Balance Sheet"):
        if balance_sheet is None or balance_sheet.empty:
            st.info("Balance Sheet not available for this ticker.")
        else:
            st.dataframe(balance_sheet)

    # st.write("SUMMARY (first 400 chars):", summary[:400])
    # st.write("SUMMARY length:", len(summary))

    # Recommendation (only if summary is valid)
    st.subheader("💡 AI Investment Recommendation")
    if isinstance(summary, str) and summary.strip() and not summary.startswith("HF Error") and not summary.startswith("HF Exception"):
        st.markdown(f"### {generate_recommendation(summary)}")
    else:
        st.warning("Recommendation skipped because AI summary is not available.")
