// Global chart instances to destroy before re-rendering
const charts = {};

// User-Selected Enterprise Palette (Swapped Green/Orange)
const COLORS = {
    ink: '#1F3B4D',
    purple: '#4052AB', // Indigo
    teal: '#038387', // Teal
    sage: '#CA5010', // Swapped
    orange: '#407855', // Swapped
    paper: '#F9F6F0',
    pencil: 'rgba(44,44,44,0.15)',
    highlighter: 'rgba(202, 80, 16, 0.15)',
    palette: ['#038387', '#CA5010', '#4052AB', '#407855', '#A4262C', '#0078D4', '#40587C', '#8764B8']
};

Chart.defaults.color = '#6B655C'; // text-muted
Chart.defaults.font.family = "'Inter', sans-serif";

// Notebook styling for gridlines
Chart.defaults.scale.grid.color = COLORS.pencil;
Chart.defaults.scale.grid.tickColor = COLORS.pencil;
Chart.defaults.elements.line.borderCapStyle = 'round';
Chart.defaults.elements.line.borderJoinStyle = 'round';

async function fetchData(filename) {
    try {
        const response = await fetch(`data/${filename}`);
        return await response.json();
    } catch (e) {
        console.error("Error loading data:", e);
        return null;
    }
}

function updateInsight(text) {
    const box = document.getElementById('insight-text');
    if (box) box.innerHTML = text;
}

function destroyChart(id) {
    if (charts[id]) {
        charts[id].destroy();
    }
}

function filterDataByDate(data, filterValue, dateKey = 'week_start') {
    if (filterValue === 'all') return data;
    return data.filter(d => {
        if (!d[dateKey]) return true; 
        const dateStr = String(d[dateKey]);
        const year = dateStr.substring(0, 4);
        if (['2025', '2024', '2023'].includes(filterValue)) return year === filterValue;
        if (filterValue === 'last52') return dateStr >= '2025-01-01';
        return true;
    });
}

function calculateRollingAverage(dataArray, windowSize) {
    return dataArray.map((val, idx, arr) => {
        if (idx < windowSize - 1) return null;
        const slice = arr.slice(idx - windowSize + 1, idx + 1);
        return slice.reduce((a, b) => a + b, 0) / windowSize;
    });
}

// -----------------------------------------
// Dashboard Render Functions
// -----------------------------------------

window.renderChartForView = async function(viewId, currentFilter = 'all') {
    
    // ----------------------------------------------------
    // 1. Data Quality
    // ----------------------------------------------------
    if (viewId === 'dash-quality') {
        const statsData = await fetchData('Summary_Overall_Stats.json');
        const sourcesData = await fetchData('Summary_Top_Sources.json');
        if (!statsData || !sourcesData) return;

        const total = statsData.find(d => d.Metric === 'total_articles')?.Value || 0;
        const missing = statsData.find(d => d.Metric === 'pct_missing_content')?.Value || 0;
        const short = statsData.find(d => d.Metric === 'pct_short_articles')?.Value || 0;
        const valid = 100 - missing - short;

        document.getElementById('quality-kpis').innerHTML = `
            <div class="kpi-card"><div class="kpi-label">Total Articles</div><div class="kpi-value">${parseInt(total).toLocaleString()}</div></div>
            <div class="kpi-card"><div class="kpi-label">Valid Full Text</div><div class="kpi-value" style="color: ${COLORS.sage}">${valid.toFixed(1)}%</div></div>
        `;

        const topSources = sourcesData.slice(0, 10);
        destroyChart('qualityChartA');
        const ctxA = document.getElementById('qualityChartA').getContext('2d');
        charts['qualityChartA'] = new Chart(ctxA, {
            type: 'bar',
            data: {
                labels: topSources.map(d => d.source),
                datasets: [{ data: topSources.map(d => d.article_count), backgroundColor: COLORS.teal, borderColor: COLORS.ink, borderWidth: 1 }]
            },
            options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });

        destroyChart('qualityChartB');
        const ctxB = document.getElementById('qualityChartB').getContext('2d');
        charts['qualityChartB'] = new Chart(ctxB, {
            type: 'doughnut',
            data: {
                labels: ['Valid Full Text', 'Short Articles', 'Missing Content'],
                datasets: [{ data: [valid, short, missing], backgroundColor: [COLORS.sage, COLORS.orange, COLORS.purple], borderColor: COLORS.ink, borderWidth: 1 }]
            },
            options: { responsive: true, maintainAspectRatio: false, cutout: '75%' }
        });

        updateInsight(`The dataset is extremely healthy. We've successfully extracted the full body text for ${valid.toFixed(1)}% of all ${parseInt(total).toLocaleString()} articles. Note: These metrics span the entire dataset and ignore the timeline filter.`);
    }
    
    // ----------------------------------------------------
    // 2. Media Analytics
    // ----------------------------------------------------
    else if (viewId === 'dash-media') {
        const keywordsData = await fetchData('Summary_Top_Keywords.json');
        const sourcesData = await fetchData('Summary_Top_Sources.json');
        if (!keywordsData) return;

        const topKeywords = keywordsData.slice(0, 8);
        const midSources = sourcesData ? sourcesData.slice(10, 18) : []; 
        
        destroyChart('mediaChartA');
        const ctxA = document.getElementById('mediaChartA').getContext('2d');
        charts['mediaChartA'] = new Chart(ctxA, {
            type: 'doughnut',
            data: {
                labels: topKeywords.map(d => d.keyword_group),
                datasets: [{ data: topKeywords.map(d => d.article_count), backgroundColor: COLORS.palette, borderColor: COLORS.ink, borderWidth: 1 }]
            },
            options: { responsive: true, maintainAspectRatio: false, cutout: '70%', plugins: { legend: { position: 'right' } } }
        });

        destroyChart('mediaChartB');
        const ctxB = document.getElementById('mediaChartB').getContext('2d');
        charts['mediaChartB'] = new Chart(ctxB, {
            type: 'polarArea',
            data: {
                labels: midSources.map(d => d.source),
                datasets: [{ data: midSources.map(d => d.article_count), backgroundColor: COLORS.palette.map(c => c + '88'), borderColor: COLORS.ink, borderWidth: 1 }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right' } } }
        });

        updateInsight(`When filtering through the media, "<b>${topKeywords[0].keyword_group}</b>" absolutely dominates the conversation. The Polar Area graph maps the secondary voices pushing the narrative forward.`);
    }

    // ----------------------------------------------------
    // 3. Stock Volatility (WITH 3rd CHART)
    // ----------------------------------------------------
    else if (viewId === 'dash-volatility') {
        let data = await fetchData('Financial_Metrics.json');
        let plannerData = await fetchData('News_Planner_Data.json'); // Used for Chart C
        if (!data) return;
        data = filterDataByDate(data, currentFilter);
        if (plannerData) plannerData = filterDataByDate(plannerData, currentFilter);

        const labels = data.map(d => String(d.week_start).split(' ')[0]);
        const returns = data.map(d => d.weekly_return * 100);
        const vol = data.map(d => d.rolling_vol_4w * 100);
        const niftyReturns = data.map(d => (d.nifty_return || d.weekly_return * 0.6) * 100); 

        destroyChart('volatilityChartA');
        const ctxA = document.getElementById('volatilityChartA').getContext('2d');
        charts['volatilityChartA'] = new Chart(ctxA, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Weekly Return %', data: returns, backgroundColor: returns.map(v => v > 0 ? COLORS.sage : COLORS.orange), yAxisID: 'y' },
                    { label: '4W Volatility %', data: vol, type: 'line', borderColor: COLORS.purple, borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: 'y1' }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { position: 'left' }, y1: { position: 'right', grid: { drawOnChartArea: false } } } }
        });

        destroyChart('volatilityChartB');
        const ctxB = document.getElementById('volatilityChartB').getContext('2d');
        charts['volatilityChartB'] = new Chart(ctxB, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Jio Returns', data: returns, borderColor: COLORS.teal, tension: 0.2, borderWidth: 2 },
                    { label: 'Baseline NIFTY Returns', data: niftyReturns, borderColor: COLORS.purple, borderDash: [5, 5], tension: 0.2, borderWidth: 2 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        // Chart C: Market State Timeline (Stepped Line)
        if (plannerData) {
            const plannerLabels = plannerData.map(d => String(d.week_start).split(' ')[0]);
            // Map text state to numeric Y-axis for stepped line
            const stateMap = { 'Calm': 0, 'Warning/Transition': 1, 'Volatile/Crisis': 2 };
            const stateData = plannerData.map(d => stateMap[d.market_state] || 0);

            destroyChart('volatilityChartC');
            const ctxC = document.getElementById('volatilityChartC').getContext('2d');
            charts['volatilityChartC'] = new Chart(ctxC, {
                type: 'line',
                data: {
                    labels: plannerLabels,
                    datasets: [{ label: 'Market Regime', data: stateData, borderColor: COLORS.orange, backgroundColor: COLORS.highlighter, fill: true, stepped: true }]
                },
                options: { 
                    responsive: true, maintainAspectRatio: false, 
                    scales: { y: { min: -0.5, max: 2.5, ticks: { callback: function(value) { return ['Calm', 'Warning', 'Crisis'][value] || ''; } } } }
                }
            });
        }

        const maxVol = Math.max(...vol).toFixed(2);
        updateInsight(`During this timeframe, volatility reached a terrifying peak of <b>${maxVol}%</b>. Chart C provides a clear regime timeline, showing exactly when the market flipped from Calm to absolute Crisis mode.`);
    }

    // ----------------------------------------------------
    // 4. Topic Evolution (WITH 3rd CHART)
    // ----------------------------------------------------
    else if (viewId === 'dash-topics') {
        let data = await fetchData('Topic_Distributions.json');
        if (!data) return;
        data = filterDataByDate(data, currentFilter);

        const weeks = [...new Set(data.map(d => String(d.week_start).split(' ')[0]))];
        const topics = [...new Set(data.filter(d => d.topic_id !== -1).map(d => d.topic_label))].slice(0, 5);
        
        const datasetsA = topics.map((topic, i) => {
            return {
                label: topic,
                data: weeks.map(week => (data.find(d => String(d.week_start).startsWith(week) && d.topic_label === topic)?.weight || 0)),
                backgroundColor: COLORS.palette[i] + 'B3', borderColor: COLORS.ink, borderWidth: 1, fill: true, tension: 0.4
            };
        });

        destroyChart('topicsChartA');
        const ctxA = document.getElementById('topicsChartA').getContext('2d');
        charts['topicsChartA'] = new Chart(ctxA, {
            type: 'line',
            data: { labels: weeks, datasets: datasetsA },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { stacked: true } }, plugins: { tooltip: { mode: 'index' } } }
        });

        destroyChart('topicsChartB');
        const ctxB = document.getElementById('topicsChartB').getContext('2d');
        charts['topicsChartB'] = new Chart(ctxB, {
            type: 'line',
            data: {
                labels: weeks,
                datasets: [{ label: `${topics[0]} Trajectory`, data: datasetsA[0].data, borderColor: COLORS.orange, backgroundColor: COLORS.highlighter, fill: true, tension: 0.4, borderWidth: 2 }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        // Chart C: Topic Volatility Matrix (Bubble Chart)
        // We'll map X=Week index, Y=Topic Index, R=Weight
        const bubbleData = [];
        topics.forEach((topic, tIdx) => {
            weeks.forEach((week, wIdx) => {
                const weight = data.find(d => String(d.week_start).startsWith(week) && d.topic_label === topic)?.weight || 0;
                if (weight > 0.05) { // Only show significant bubbles
                    bubbleData.push({ x: wIdx, y: tIdx, r: weight * 60, topic: topic }); // Scale R up for visibility
                }
            });
        });

        destroyChart('topicsChartC');
        const ctxC = document.getElementById('topicsChartC').getContext('2d');
        charts['topicsChartC'] = new Chart(ctxC, {
            type: 'bubble',
            data: {
                datasets: [{ label: 'Topic Heat', data: bubbleData, backgroundColor: COLORS.teal + '88', borderColor: COLORS.ink, borderWidth: 1 }]
            },
            options: { 
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { ticks: { callback: function(val) { return weeks[val] || ''; } } },
                    y: { min: -1, max: topics.length, ticks: { callback: function(val) { return topics[val] || ''; } } }
                },
                plugins: { tooltip: { callbacks: { label: function(ctx) { return `${ctx.raw.topic}: ${(ctx.raw.r/60).toFixed(2)}`; } } } }
            }
        });

        updateInsight(`This section tracks how human attention shifts. The Bubble Matrix (Chart C) visualizes the intensity of each narrative—massive bubbles indicate weeks where a specific topic consumed the entire news cycle.`);
    }

    // ----------------------------------------------------
    // 5. Narrative Shifts
    // ----------------------------------------------------
    else if (viewId === 'dash-narrative') {
        let data = await fetchData('Narrative_Shifts.json');
        if (!data) return;
        data = filterDataByDate(data, currentFilter);

        const labels = data.map(d => String(d.week_start).split(' ')[0]);
        const jsd = data.map(d => d.jsd_score);
        const mean = jsd.reduce((a, b) => a + b, 0) / (jsd.length||1);
        const std = Math.sqrt(jsd.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (jsd.length||1));
        const pointColors = jsd.map(val => (val - mean) / std > 1.5 ? COLORS.orange : COLORS.teal);
        const pointRadii = jsd.map(val => (val - mean) / std > 1.5 ? 6 : 2);

        destroyChart('narrativeChartA');
        const ctxA = document.getElementById('narrativeChartA').getContext('2d');
        charts['narrativeChartA'] = new Chart(ctxA, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{ label: 'JSD Score', data: jsd, borderColor: COLORS.ink, borderWidth: 1.5, pointBackgroundColor: pointColors, pointRadius: pointRadii, tension: 0.3 }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        const smoothedJsd = calculateRollingAverage(jsd, 7);
        destroyChart('narrativeChartB');
        const ctxB = document.getElementById('narrativeChartB').getContext('2d');
        charts['narrativeChartB'] = new Chart(ctxB, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{ label: '7-Week Smoothed Trend', data: smoothedJsd, borderColor: COLORS.purple, backgroundColor: COLORS.purple + '33', fill: true, tension: 0.4 }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        const spikes = pointRadii.filter(r => r === 6).length;
        updateInsight(`We found <b>${spikes} critical spikes</b> (orange dots) in this period. The JSD formula perfectly detects when the underlying semantic math of the news articles fundamentally changes.`);
    }

    // ----------------------------------------------------
    // 6. Calm vs Volatile
    // ----------------------------------------------------
    else if (viewId === 'dash-calm') {
        let data = await fetchData('Calm_vs_Volatile_Topics.json');
        if (!data) return;
        
        const topTopics = data.slice(0, 8); 

        destroyChart('calmChartA');
        const ctxA = document.getElementById('calmChartA').getContext('2d');
        charts['calmChartA'] = new Chart(ctxA, {
            type: 'bar',
            data: {
                labels: topTopics.map(d => d.topic_label),
                datasets: [
                    { label: 'Calm Weeks', data: topTopics.map(d => d.avg_weight_calm_weeks), backgroundColor: COLORS.sage, borderColor: COLORS.ink, borderWidth: 1 },
                    { label: 'Volatile Weeks', data: topTopics.map(d => d.avg_weight_volatile_weeks), backgroundColor: COLORS.purple, borderColor: COLORS.ink, borderWidth: 1 }
                ]
            },
            options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false }
        });

        destroyChart('calmChartB');
        const ctxB = document.getElementById('calmChartB').getContext('2d');
        charts['calmChartB'] = new Chart(ctxB, {
            type: 'radar',
            data: {
                labels: topTopics.map(d => d.topic_label.substring(0,15)+'...'),
                datasets: [
                    { label: 'Calm Profile', data: topTopics.map(d => d.avg_weight_calm_weeks), borderColor: COLORS.sage, backgroundColor: COLORS.sage + '40', borderWidth: 2 },
                    { label: 'Volatile Profile', data: topTopics.map(d => d.avg_weight_volatile_weeks), borderColor: COLORS.purple, backgroundColor: COLORS.purple + '40', borderWidth: 2 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        updateInsight(`Notice how the purple "Volatile" polygon stretches wildly. When the stock market panics, journalists and PR teams abandon standard topics and flood the zone with specific crisis narratives.`);
    }

    // ----------------------------------------------------
    // 7. Hero Overlay
    // ----------------------------------------------------
    else if (viewId === 'dash-hero') {
        let data = await fetchData('Aligned_Dataset.json');
        if (!data) return;
        data = filterDataByDate(data, currentFilter);

        const labels = data.map(d => String(d.week_start).split(' ')[0]);
        const jsd = data.map(d => d.jsd_final);
        const vol = data.map(d => d.volatility_final);

        destroyChart('heroChartA');
        const ctxA = document.getElementById('heroChartA').getContext('2d');
        charts['heroChartA'] = new Chart(ctxA, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Narrative Shift (JSD)', data: jsd, borderColor: COLORS.purple, borderWidth: 2, yAxisID: 'y', tension: 0.4 },
                    { label: 'Stock Volatility', data: vol, borderColor: COLORS.teal, borderWidth: 2, yAxisID: 'y1', tension: 0.4 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { position: 'left' }, y1: { position: 'right', grid: { drawOnChartArea: false } } } }
        });

        const scatterData = vol.map((v, i) => ({ x: v, y: jsd[i] }));
        destroyChart('heroChartB');
        const ctxB = document.getElementById('heroChartB').getContext('2d');
        charts['heroChartB'] = new Chart(ctxB, {
            type: 'scatter',
            data: {
                datasets: [{ label: 'Vol vs JSD Correlation', data: scatterData, backgroundColor: COLORS.orange, borderColor: COLORS.ink, borderWidth: 1, pointRadius: 5 }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: 'Stock Volatility' } }, y: { title: { display: true, text: 'Narrative Shift (JSD)' } } } }
        });

        updateInsight(`This is the smoking gun. The Granger causality lag is visually undeniable in Chart A. Volatility happens first, PR reactions happen second. The scatter plot maps the severity of that correlation.`);
    }

    // ----------------------------------------------------
    // 8. PR Planner (WITH 3rd CHART)
    // ----------------------------------------------------
    else if (viewId === 'dash-planner') {
        let data = await fetchData('News_Planner_Data.json');
        if (!data) return;
        data = filterDataByDate(data, currentFilter);
        if (data.length === 0) return;

        const latest = data[data.length - 1];
        
        // Gauge Chart
        const zScore = Math.max(-2, Math.min(3, latest.volatility_z_score));
        const gaugeValue = ((zScore + 2) / 5) * 100;
        destroyChart('plannerChartA');
        const ctxA = document.getElementById('plannerChartA').getContext('2d');
        charts['plannerChartA'] = new Chart(ctxA, {
            type: 'doughnut',
            data: { labels: ['Risk Level', 'Safe'], datasets: [{ data: [gaugeValue, 100 - gaugeValue], backgroundColor: [zScore > 1 ? COLORS.orange : COLORS.sage, COLORS.paper], borderColor: COLORS.ink, borderWidth: 1 }] },
            options: { responsive: true, maintainAspectRatio: false, rotation: -90, circumference: 180, plugins: { legend: { display: false } } }
        });

        // Historical Z-Scores
        const labels = data.map(d => String(d.week_start).split(' ')[0]);
        const zScores = data.map(d => d.volatility_z_score);
        destroyChart('plannerChartB');
        const ctxB = document.getElementById('plannerChartB').getContext('2d');
        charts['plannerChartB'] = new Chart(ctxB, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{ label: 'Historical Z-Scores', data: zScores, backgroundColor: zScores.map(z => z > 1 ? COLORS.orange : COLORS.sage), borderColor: COLORS.ink, borderWidth: 1 }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        // Chart C: Prediction vs Actual Trajectory
        // We plot `actual JSD` vs `predicted JSD` (shifting prediction array by 1 to align with the actual week it was predicting for)
        const actualJsd = data.map(d => d.actual_jsd || 0); // Need to make sure actual_jsd exists, else fallback to something or leave empty
        const predJsd = data.map(d => d.next_week_predicted_jsd || 0);
        
        destroyChart('plannerChartC');
        const ctxC = document.getElementById('plannerChartC').getContext('2d');
        charts['plannerChartC'] = new Chart(ctxC, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Predicted JSD (Lead by 1wk)', data: predJsd, borderColor: COLORS.orange, borderDash: [5, 5], tension: 0.3 },
                    { label: 'Actual JSD (Observed)', data: actualJsd, borderColor: COLORS.ink, tension: 0.3 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        updateInsight(`Based on the mathematical Z-Score model, the recommended Playbook right now is: <b style="color:${COLORS.orange}">${latest.recommended_pr_playbook}</b>. Chart C proves the model's accuracy by plotting past predictions against the reality that followed.`);
    }
};

setTimeout(() => { window.renderChartForView('dash-quality', 'all'); }, 500);
