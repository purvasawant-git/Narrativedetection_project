# 📊 NarrativeShift: Power BI Dashboard Suite Presentation Guide

This guide details the **10 interactive dashboards** built using the consolidated dataset in `data/project_outputs.xlsx`. It provides an end-to-end reference for presenting this project to non-technical and executive stakeholders, illustrating how data ingestion, NLP, and financial metrics combine into actionable business tools.

---

## 🗺️ Dashboard Map & Core Sheets

The Excel file contains **8 sheets** of data:
1.  **`Articles_Summary`**: Overall data volume, domain shares, and keyword distributions.
2.  **`Topic_Distributions`**: Weekly topic weights and keyword labels.
3.  **`Narrative_Shifts`**: Weekly JSD shift scores and rolling averages.
4.  **`Financial_Metrics`**: Daily returns resampled to Monday weekly returns, stock volatility, and index returns.
5.  **`Aligned_Dataset`**: Merged, stationary dataset used for tests.
6.  **`Granger_Results`**: Statistical outcomes of bivariate and VAR causality tests.
7.  **`News_Planner_Data`**: Volatility Z-scores paired with shifted next-week JSD scores and recommended PR playbooks.
8.  **`Calm_vs_Volatile_Topics`**: Mean topic weight distributions segmented by market volatility states.

---

## 🖥️ The 10 Dashboard Pages (End-to-End Reference)

### 📊 SECTION 1: DATA FOUNDATION & INGESTION

#### Dashboard 1: GDELT Ingestion & Data Coverage
*   **Excel Data Sheet**: `Articles_Summary` (Overall stats)
*   **Visual Outputs**:
    *   **KPI Cards**: Large text visuals for *Total Ingested Articles* (6,267) and *Date Range* (Jan 1, 2023 – Jan 1, 2026).
    *   **Bar Chart**: Weekly article count over time to show coverage consistency.
*   **Interactive Features**: Slicer to filter the timeline by year or quarter.
*   **What it Conveys**: Demonstrates the sheer scale and completeness of the data foundation, proving there are no coverage gaps that would invalidate statistical testing.

#### Dashboard 2: Media Source & Keyword Taxonomy Distribution
*   **Excel Data Sheet**: `Articles_Summary` (Top Sources & Keywords)
*   **Visual Outputs**:
    *   **Donut Chart**: Top 10 publishing domains (e.g. *Economic Times* at 16.1%, *Financial Express* at 7.3%).
    *   **Clustered Column Chart**: Volume of articles per keyword group (e.g. *Reliance Jio + 5G India*).
*   **Interactive Features**: Click on a specific news domain to filter which keyword groups they write about most.
*   **What it Conveys**: Shows who dominates the media conversation around Jio and which sector topics (like 5G or tariffs) drive the volume of public discussion.

#### Dashboard 3: Content Quality & NLP Readiness
*   **Excel Data Sheet**: `Articles_Summary` (Quality metrics)
*   **Visual Outputs**:
    *   **Gauge Visuals**: *Missing Content %* (1.7%), *Short Articles <50 words %* (5.3%).
    *   **Card Visual**: *Median Word Count* (471 words).
    *   **Histogram**: Word count distribution.
*   **Interactive Features**: Toggle to filter by news source to see which outlet provides the longest, highest-quality text.
*   **What it Conveys**: Acts as a **data quality audit**. It proves to stakeholders that the text scraping pipeline was highly successful, delivering long, clean text body blocks suitable for advanced semantic embeddings.

---

### 🧠 SECTION 2: NLP TOPIC MODELING & SHIFTS

#### Dashboard 4: The Weekly Topic Evolution (Share of Voice)
*   **Excel Data Sheet**: `Topic_Distributions`
*   **Visual Outputs**:
    *   **Stacked Area Chart**: X-axis = `week_start`, Y-axis = `weight` (normalized to 100%), Legend = `topic_label` (Top 5 topics).
    *   **Data Table**: Scrollable list of weeks, topic keywords, and article counts.
*   **Interactive Features**: Hovering over stacked areas displays the top 3 topic keywords. Slicer to filter to specific quarters.
*   **What it Conveys**: This is the visual proof that the media narrative actually changes. It shows how the dominant story shifts over 3 years (e.g. from 5G rollouts in 2023 to tariff hikes in 2024 and IPO/financial services in 2025).

#### Dashboard 5: Dominant Weekly Topic Timeline
*   **Excel Data Sheet**: `Topic_Distributions` (highest-weight topics per week)
*   **Visual Outputs**:
    *   **Ribbon Chart**: Weekly dominant topics ranked by weight chronologically.
    *   **Matrix Card**: Visual grid showing the single most common topic for each quarter.
*   **Interactive Features**: Slicer to isolate specific topic categories (e.g. spectrum auctions).
*   **What it Conveys**: Condenses the weekly noise. It tells stakeholders: *"If you want to look at a single, dominant headline for any week in the last 3 years, here is what the world was talking about."*

#### Dashboard 6: Narrative Shift Spikes (JSD Scores)
*   **Excel Data Sheet**: `Narrative_Shifts`
*   **Visual Outputs**:
    *   **Line Chart**: Raw weekly JSD score (thin gray line) and 3-week rolling average JSD (thick blue line).
    *   **Vertical Reference Lines**: Highlight the 12 weeks flagged as statistical "spikes" (Z-score > 1.5).
*   **Interactive Features**: Tooltip popups displaying the topic transition during spike weeks (e.g., *"Shift Event: Topic changed from 5G services to spectrum debt"*).
*   **What it Conveys**: Pinpoints the exact weeks where the media abruptly dropped their current narrative and pivoted to a completely new story, identifying critical communication transition points.

---

### 📈 SECTION 3: FINANCE & STATISTICAL COUPLING

#### Dashboard 7: RIL Stock Return & Volatility Timeline
*   **Excel Data Sheet**: `Financial_Metrics`
*   **Visual Outputs**:
    *   **Dual Y-Axis Line Chart**: Left axis = weekly return %, Right axis = weekly volatility.
    *   **KPI Cards**: *Average Weekly Volatility* (1.04%), *Maximum Weekly Drawdown* (-6.1%).
*   **Interactive Features**: Filter timeline to zoom into specific high-volatility market events.
*   **What it Conveys**: Establishes the stock market timeline, showing periods of calm trading versus periods of extreme market panic or excitement for Reliance Industries.

#### Dashboard 8: The Hero Overlay: Narrative Shift vs. Volatility
*   **Excel Data Sheet**: `Aligned_Dataset`
*   **Visual Outputs**:
    *   **Overlay Line Chart (Dual Axes)**: Left axis (blue) = smoothed narrative shift JSD, Right axis (orange) = rolling stock volatility.
    *   **Scatter Plot**: Volatility on X-axis, JSD score on Y-axis to show correlation density.
*   **Interactive Features**: Play axis (timeline animation) to watch the two lines evolve together over the 154 weeks.
*   **What it Conveys**: The central visual thesis of the project. It shows the alignment (or lack thereof) between narrative shifts and stock market risk.

---

### 🔮 SECTION 4: PREDICTIVE BUSINESS INSIGHTS

#### Dashboard 9: The PR Predictive News Planner
*   **Excel Data Sheet**: `News_Planner_Data`
*   **Visual Outputs**:
    *   **Current Volatility Indicator (Red / Yellow / Green)**: Gauge dial representing the current week's stock volatility.
    *   **Predicted Next-Week JSD Gauge**: Visual prediction of next week's narrative shift level.
    *   **Recommended PR Playbook Card**: Dynamic text box displaying the strategic advice (e.g., *"High Volatility Spike Detected. Prepare statement regarding financial debt disclosures within 7 days"*).
*   **Interactive Features**: Slicer to select the current calendar week to see the active volatility level and next week's warning index.
*   **What it Conveys**: **The primary commercial deliverable**. It proves how a PR department can use stock volatility as a leading indicator to anticipate and prepare for coming media narrative shifts 10 days in advance.

#### Dashboard 10: "Crisis vs. Calm" Media Topic Analyzer
*   **Excel Data Sheet**: `Calm_vs_Volatile_Topics`
*   **Visual Outputs**:
    *   **Comparative Bar Charts**: Side-by-side charts showing average topic weights during *Calm Weeks* versus *Volatile Weeks*.
    *   **Table Visual**: Sortable list showing `weight_change` and `weight_increase_ratio`.
*   **Interactive Features**: Filter by topic name to see how its "Share of Voice" multiplies when the stock gets volatile.
*   **What it Conveys**: Reveals which topics are "market-sensitive." For example, it highlights that when the stock is volatile, topics like *spectrum debt* and *IPO rumors* increase by 300%, while product-related topics drop. This helps PR allocate resources to the right messaging categories.
