# Phase II Roadmap: Transitioning to Predictive Stock Forecasting

This document serves as a strategic and technical guide for expanding the **Jio Narrative Shift Detection** pipeline into a predictive trading and warning engine. 

---

## 🔍 Retrospective: Phase I Findings
Our Phase I analysis established the following:
1.  **Market Efficiency (News $\rightarrow$ Stock)**: Media narrative shifts do not Granger-cause stock price volatility at a weekly level ($p > 0.59$). This is consistent with a semi-strong form efficient market that absorbs public information rapidly, erasing weekly lead-lag opportunities.
2.  **Reactive Journalism (Stock $\rightarrow$ News)**: Stock price volatility Granger-causes media narrative shifts **1 to 2 weeks later** (lag 1: $p = 0.0441$, lag 2: $p = 0.0334$). Stock prices move first, and the media shifts its coverage focus retroactively to explain the movement.

**Current Deliverable**: A **PR & Corporate Communications Early-Warning System**. This allows PR teams to monitor stock volatility and anticipate changes in media coverage topics 10 days in advance.

---

## 🚀 Phase II: Rebuilding for Daily Forecasting

To transition this framework from a descriptive tool into a daily predictive trading engine, the following upgrades are proposed:

### 1. Daily Temporal Granularity
*   **Methodology**: Aggregate all news articles and stock metrics by **day** instead of by **ISO week**.
*   **Implementation**:
    *   Modify `preprocessing.py` to group articles by date (`YYYY-MM-DD`).
    *   Compute daily JSD scores ($JSD_{t, t-1}$) across consecutive days.
    *   Measure daily stock returns and daily **Parkinson Volatility** (which uses High/Low intraday prices to capture daily trading range):
        $$\text{Parkinson Volatility} = \sqrt{\frac{1}{4 \ln 2} \ln\left(\frac{\text{High}_t}{\text{Low}_t}\right)^2}$$

### 2. Semantic Sentiment Integration (FinBERT)
*   **Methodology**: Pure topic changes do not indicate price direction. We must score whether a narrative shift is positive or negative.
*   **Implementation**:
    *   Run a pretrained financial sentiment classifier (e.g. `yiyanghkust/finbert-tone` from Hugging Face) on all ingested article texts.
    *   Calculate a daily **Net Sentiment Index**:
        $$\text{Net Sentiment}_t = \frac{\text{Positive Articles}_t - \text{Negative Articles}_t}{\text{Total Articles}_t}$$
    *   Scale the JSD score by the sentiment direction to get a "Directional Narrative Shift Index."

### 3. Media Reach & Source Weighting
*   **Methodology**: Mainstream publications (e.g. *The Economic Times*, *Reuters*) have more market impact than small blogs.
*   **Implementation**:
    *   Map domain names to reach/credibility weightings (e.g., tier-1 media = 5.0, tier-3 blogs = 1.0).
    *   Incorporate GDELT article counts or volume indicators to weight the weekly topic probability matrices.

### 4. Non-Linear Machine Learning Classification
*   **Methodology**: Granger Causality is a linear test. Markets behave non-linearly.
*   **Implementation**:
    *   Train classification models (e.g., **XGBoost**, **Random Forest**, or an **LSTM** neural network).
    *   **Features (Inputs)**: Daily JSD score, daily Net Sentiment, 3-day rolling JSD, technical indicators (RSI, MACD, Volume), and NIFTY 50 returns.
    *   **Target (Output)**: Next-day stock direction (1 if return $> 0$, else 0) or next-day volatility spike (1 if volatility $> 90\text{th percentile}$, else 0).
