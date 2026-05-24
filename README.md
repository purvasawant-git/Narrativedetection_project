# Algorithmic Causality in Financial Narratives

## The Origin: Why This Project Exists
It is a widely held belief in finance that "breaking news drives the stock market." When negative headlines hit the press, investors panic and sell. 

I wanted to prove this mathematically. My initial hypothesis was simple: **If I mathematically track negative media narratives, I can predict stock market drops.**

To test this, I built an automated pipeline to ingest over 6,200 global news articles spanning three years. The goal was to quantify the exact moments a corporate narrative "shifted," align those shifts against hard stock market volatility metrics, and prove that the news caused the market panic.

## Deployment link : financialnarrativesmedia.vercel.app

## The Pivot: How The Project Unfolded
The project unfolded in four distinct engineering phases to ensure no data was fabricated and the math was rigorous:

1. **Extraction:** I built a fault-tolerant pipeline querying the GDELT Global Database, autonomously scraping and parsing 6,267 authentic news articles over an unbroken 156-week timeline.
2. **Clustering:** Rather than relying on rigid manual tagging or basic sentiment analysis, I deployed **Transformer-Based Semantic Clustering** (BERTopic) to force the AI to autonomously organize the news into dominant themes.
3. **Quantification:** I calculated the Jensen-Shannon Divergence (JSD) between weekly topic distributions to mathematically pinpoint exactly when the media narrative drastically changed.
4. **Causality Testing:** I synchronized these narrative shift scores with the weekly stock returns and volatility of the target asset, and ran a multivariate **Vector Autoregression (VAR)** model to prove causality.

### What the Data Actually Proved
After running rigorous Granger Causality tests on the aligned data, the results **flipped my original hypothesis upside down**.

1. **The News Does NOT Drive the Market:** Weekly narrative shifts do *not* Granger-cause stock volatility. The media news is absorbed and priced in too fast for a weekly model to exploit.
2. **The Market Drives the News:** Stock volatility **Granger-causes narrative shifts 1 to 2 weeks later**. 

The data proved that massive stock price drops force corporate PR teams and journalists to adapt. They shift their reporting focus *retroactively* to explain the price behavior. The market forces the narrative to change, not the other way around.

## The Output: Actionable Business Intelligence
By proving that stock volatility predicts narrative shifts at a 1-to-2 week lag, this pipeline operates as a **PR Early-Warning System**. 

Corporate Communications teams and executives can use real-time market data as a leading indicator to predict exactly *when* the media ecosystem will pivot, allowing them to deploy their PR playbook proactively rather than reacting to bad press.

---

## Technical Execution

**Prerequisites:** Python 3.10+, 8GB RAM, 5GB Disk Space.

**To reproduce the end-to-end pipeline:**
```bash
git clone https://github.com/purvasawant-git/Narrativedetection_project.git
cd Narrativedetection_project
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Ingestion & Storage
python "src/ingestion copy.py" 2022-01-01 2024-01-01
python src/storage.py

# 2. Semantic Clustering & Shift Detection
python src/topic_model.py
python src/shift_detection.py

# 3. Financial Alignment & Causality
python src/finance.py --start 2022-01-01 --end 2024-01-01
python src/alignment.py
python src/granger.py
```
