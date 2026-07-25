\# Stock Market Intelligence System



An end-to-end ML system that predicts next-day stock direction for 5 Indian assets (Sensex, Reliance, TCS, Infosys, HDFC Bank), explains predictions with SHAP, and answers questions about market news using a RAG pipeline.



\## Features

\- Random Forest classifiers trained per asset with time-based train/test splits

\- SHAP explainability (global feature importance + per-prediction breakdown)

\- RAG-powered chat using Sentence Transformers, ChromaDB, and Cohere

\- Interactive Streamlit app with Predictions, Chat, and Comparison tabs



\## Tech Stack

Python, scikit-learn, SHAP, Sentence Transformers, ChromaDB, Cohere, Streamlit, Plotly, yfinance



\## Setup

1\. Clone this repo

2\. `pip install -r requirements.txt`

3\. Create `.streamlit/secrets.toml` with your own `COHERE\_API\_KEY = "your\_key"`

4\. `streamlit run app.py`



\## Screenshots

!\[Predictions Tab](screenshots/predictions.png)

!\[Chat Tab](screenshots/chat.png)

!\[Comparison Tab](screenshots/comparison.png)



\## Key Findings

\- Only the Sensex model showed AUC meaningfully above random (0.548); individual stocks were near or below random, reflecting the difficulty of short-horizon stock prediction from price data alone

\- Threshold tuning based on business costs reduced total cost across all 5 assets



\## Author

Kripa

