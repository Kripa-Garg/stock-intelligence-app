import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
from sentence_transformers import SentenceTransformer
import chromadb
import cohere
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

st.set_page_config(
    page_title="Stock Market Intelligence System", layout="wide")
st.title("📈 Stock Market Intelligence System")

assets = ['^BSESN', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']


@st.cache_resource
def load_everything():
    stock_data = {}
    models = {}
    for ticker in assets:
        stock_data[ticker] = pd.read_csv(
            f'data_{ticker}.csv', index_col=0, parse_dates=True)
        models[ticker] = joblib.load(f'model_{ticker}.pkl')
    news_df = pd.read_csv('news_data.csv')
    return stock_data, models, news_df


stock_data, models, news_df = load_everything()

feature_cols = [
    'daily_return', 'log_return', 'high_low_ratio',
    'ma_7', 'ma_14', 'ma_30', 'price_to_ma7', 'ma_crossover',
    'volatility_7', 'volatility_14',
    'day_of_week', 'month', 'quarter', 'is_monday', 'is_friday',
    'volume_ma_7', 'volume_ratio'
]

# ---- Set up RAG pipeline once, cached ----


@st.cache_resource
def setup_rag(news_df):
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    news_df = news_df.dropna(subset=['title', 'description']).drop_duplicates(
        subset=['title']).reset_index(drop=True)
    news_df['full_text'] = news_df['title'].fillna(
        '') + ". " + news_df['description'].fillna('')
    documents = news_df['full_text'].tolist()
    doc_ids = news_df.index.astype(str).tolist()

    embeddings = embed_model.encode(documents)

    client = chromadb.Client()
    collection = client.get_or_create_collection("news")
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])
    collection.add(
        ids=doc_ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=news_df[['ticker', 'company', 'publishedAt']].astype(
            str).to_dict('records')
    )
    return embed_model, collection


embed_model, collection = setup_rag(news_df)

co = cohere.Client(st.secrets["COHERE_API_KEY"])


def search_news(query, n_results=3, ticker_filter=None):
    query_embedding = embed_model.encode([query])
    where_clause = {"ticker": ticker_filter} if ticker_filter else None
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        where=where_clause
    )
    return results


def answer_question(query, ticker_filter=None):
    retrieved = search_news(query, n_results=3, ticker_filter=ticker_filter)
    context_articles = retrieved['documents'][0]

    if not context_articles:
        return "I couldn't find relevant news articles to answer that."

    context_text = "\n\n".join(context_articles)
    prompt = f"""Based on the following news articles, answer the question concisely.

Articles:
{context_text}

Question: {query}

Answer:"""

    response = co.chat(
        message=prompt, model="command-a-03-2025", temperature=0.3)
    return response.text


# ---- Sidebar ----
st.sidebar.header("Settings")
selected_ticker = st.sidebar.selectbox("Choose an asset", assets)

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "💬 Chat", "⚖️ Comparison"])

with tab1:
    st.header(f"Prediction for {selected_ticker}")
    df = stock_data[selected_ticker]
    model = models[selected_ticker]

    fig = px.line(df, x=df.index, y='Close',
                  title=f'{selected_ticker} Price Trend')
    st.plotly_chart(fig, width='stretch')

    latest_row = df[feature_cols].iloc[[-1]]
    prediction = model.predict(latest_row)[0]
    confidence = model.predict_proba(latest_row)[0][prediction]

    col1, col2 = st.columns(2)
    with col1:
        direction = "🟢 UP" if prediction == 1 else "🔴 DOWN"
        st.metric("Predicted Direction (Tomorrow)", direction)
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    st.subheader("Why this prediction?")

    classifier = model.named_steps['classifier']
    scaler = model.named_steps['scaler']

    X_recent = df[feature_cols].iloc[-100:]
    X_recent_scaled = scaler.transform(X_recent)
    X_recent_scaled_df = pd.DataFrame(
        X_recent_scaled, columns=X_recent.columns, index=X_recent.index)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_recent_scaled_df)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Feature Importance (overall)**")
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': abs(shap_values[:, :, 1]).mean(axis=0)
        }).sort_values('importance', ascending=True)
        fig_importance = px.bar(
            importance_df, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig_importance, width='stretch')

    with col2:
        st.write("**Today's Prediction Breakdown**")
        latest_scaled = scaler.transform(latest_row)
        latest_explanation = explainer.shap_values(latest_scaled)
        waterfall_df = pd.DataFrame({
            'feature': feature_cols,
            'shap_value': latest_explanation[0, :, 1]
        }).sort_values('shap_value', key=abs, ascending=True)
        fig_waterfall = px.bar(waterfall_df, x='shap_value', y='feature', orientation='h',
                               color='shap_value', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_waterfall, width='stretch')

with tab2:
    st.header("Ask about the news")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    user_question = st.chat_input(
        "Ask something like: What's the sentiment around Reliance?")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching news and generating answer..."):
                answer = answer_question(user_question)
                st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))

with tab3:
    st.header("Compare All Assets")

    @st.cache_data
    def compute_comparison():
        rows = []
        predictions = []
        for ticker in assets:
            df = stock_data[ticker]
            model = models[ticker]

            X = df[feature_cols]
            y = df['target']
            split_index = int(len(df) * 0.8)
            X_test = X.iloc[split_index:]
            y_test = y.iloc[split_index:]

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            rows.append({
                'ticker': ticker,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba)
            })

            latest_row = X.iloc[[-1]]
            pred = model.predict(latest_row)[0]
            conf = model.predict_proba(latest_row)[0][pred]
            predictions.append({
                'ticker': ticker,
                'prediction': "UP" if pred == 1 else "DOWN",
                'confidence': f"{conf:.1%}"
            })

        return pd.DataFrame(rows).round(3), pd.DataFrame(predictions)

    metrics_df, predictions_df = compute_comparison()

    st.subheader("Model Performance Comparison")
    st.dataframe(metrics_df, width='stretch')

    fig_metrics = px.bar(metrics_df, x='ticker', y=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                         barmode='group', title='Metrics by Asset')
    st.plotly_chart(fig_metrics, width='stretch')

    st.subheader("Current Predictions — All Assets")
    st.dataframe(predictions_df, width='stretch')
