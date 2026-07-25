# 📈 Stock Market Intelligence System

An end-to-end machine learning system that predicts next-day stock direction for major Indian market assets, explains its predictions with SHAP, and answers natural-language questions about market news through a Retrieval-Augmented Generation (RAG) pipeline — all wrapped in an interactive Streamlit application.

Built as a capstone project simulating the role of a data scientist at a quantitative investment firm.

---

## 🎯 Overview

This system combines classical ML with modern LLM techniques to deliver a full investment-research workflow:

- **Predict** — Random Forest classifiers forecast next-day UP/DOWN movement for 5 assets
- **Explain** — SHAP values break down *why* the model made each prediction, both globally and per-prediction
- **Ask** — A RAG pipeline answers natural-language questions about recent market news, grounded in real retrieved articles
- **Compare** — Side-by-side performance metrics across all assets

It can answer questions like:
> *"What is the predicted direction for Reliance tomorrow?"*
> *"Which stock has the highest accuracy?"*
> *"Explain why Sensex is predicted to go UP."*
> *"Compare the performance of TCS and Infosys."*

---

## 🏗️ Architecture
Data Layer → yfinance (5yr price history) + NewsAPI (headlines)
Feature Engineering → 23 engineered features (returns, moving averages,
volatility, volume, time-based signals)
Modeling → Random Forest classifiers, one per asset,
time-based train/test split (no lookahead bias)
Explainability → SHAP TreeExplainer (global + per-prediction)
Retrieval → Sentence Transformers (MiniLM) embeddings → ChromaDB
Generation → Cohere Command chat model, grounded on retrieved articles
Interface → Streamlit (3-tab web app)
