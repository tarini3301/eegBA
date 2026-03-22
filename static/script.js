/**
 * NeuroAge — Brain Age Prediction Frontend
 * Multi-Model Edition
 * ==========================================
 * Handles: form building, sample loading, API calls,
 * gauge rendering, SHAP charts, importance charts,
 * recommendation display, and model selection.
 */

// ═══════════════ CONFIGURATION ═══════════════

const REGIONS = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"];
const BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"];

const FEATURE_DEFAULTS = {};
const FEATURE_DISPLAY = {};
const FEATURE_UNITS = {};

REGIONS.forEach(region => {
    BANDS.forEach(band => {
        const fname = `${region}_${band}_Power`;
        
        let min = 0.01, max = 150, step = 'any', placeholder = '20.0', icon = '⚡';
        
        if (band === 'Delta') { max = 100; placeholder = '25.0'; icon = '🌊'; }
        else if (band === 'Theta') { max = 60; placeholder = '15.0'; icon = '🧘'; }
        else if (band === 'Alpha') { max = 90; placeholder = '30.0'; icon = '⚡'; }
        else if (band === 'Beta') { max = 50; placeholder = '12.0'; icon = '🎯'; }
        else if (band === 'Gamma') { max = 30; placeholder = '5.0'; icon = '🧠'; }

        FEATURE_DEFAULTS[fname] = { min, max, step, placeholder, icon };
        FEATURE_DISPLAY[fname] = `${region} ${band} Power`;
        FEATURE_UNITS[fname] = 'µV²';
    });
});


// ═══════════════ INITIALIZATION & STATE ═══════════════

let sampleData = [];
let availableModels = [];
let activeModelKey = 'ensemble';

document.addEventListener('DOMContentLoaded', () => {
    buildFeatureInputs();
    loadModels();
    loadSampleSubjects();
    setupFormHandlers();
});


function buildFeatureInputs() {
    const grid = document.getElementById('features-grid');
    grid.innerHTML = '';

    for (const [fname, config] of Object.entries(FEATURE_DEFAULTS)) {
        const card = document.createElement('div');
        card.className = 'feature-input-card';
        const unit = FEATURE_UNITS[fname] ? ` (${FEATURE_UNITS[fname]})` : '';
        card.innerHTML = `
            <label for="${fname}">
                <span class="label-icon">${config.icon}</span>
                ${FEATURE_DISPLAY[fname]}${unit}
            </label>
            <input type="number" id="${fname}" name="${fname}"
                   min="${config.min}" max="${config.max}" step="${config.step}"
                   placeholder="${config.placeholder}"
                   class="input-field" required>
        `;
        grid.appendChild(card);
    }
}


// ═══════════════ API LOADERS ═══════════════

async function loadModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        availableModels = data.models || [];
        renderModelSelector();
    } catch (e) {
        console.error('Failed to load models:', e);
        document.getElementById('model-selector').innerHTML =
            '<span style="color:var(--text-muted);font-size:0.82rem;">Could not load models. Server may be offline.</span>';
    }
}

async function loadSampleSubjects() {
    try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        sampleData = data.samples || [];
        renderSampleButtons();
    } catch (e) {
        console.error('Failed to load samples:', e);
        document.getElementById('sample-buttons').innerHTML =
            '<span style="color:var(--text-muted);font-size:0.82rem;">Could not load samples</span>';
    }
}


// ═══════════════ UI RENDERING: SELECTORS ═══════════════

function renderModelSelector() {
    const container = document.getElementById('model-selector');
    container.innerHTML = '';

    availableModels.forEach(model => {
        const btn = document.createElement('div');
        btn.className = `model-option ${model.key === activeModelKey ? 'active' : ''}`;
        
        let scoreStr = '';
        if (model.key !== 'ensemble') {
            scoreStr = `<div class="model-option-score">R²: ${model.r2.toFixed(3)} | MAE: ${model.mae.toFixed(1)}y</div>`;
        } else {
            scoreStr = `<div class="model-option-score" style="color:var(--green)">Best Overall Performance</div>`;
        }

        btn.innerHTML = `
            <div class="model-option-name">${model.name}</div>
            <div class="model-option-desc">${model.description}</div>
            ${scoreStr}
        `;
        
        btn.addEventListener('click', () => {
            activeModelKey = model.key;
            document.querySelectorAll('.model-option').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
        
        container.appendChild(btn);
    });
}

function renderSampleButtons() {
    const container = document.getElementById('sample-buttons');
    if (sampleData.length === 0) {
        container.innerHTML = '<span style="color:var(--text-muted);font-size:0.82rem;">No samples available</span>';
        return;
    }

    container.innerHTML = '';
    sampleData.forEach((s, i) => {
        const btn = document.createElement('button');
        btn.className = 'sample-btn';
        btn.type = 'button';
        btn.textContent = `${s.subject_id} (${s.gender}, ${Math.round(s.chronological_age)}y)`;
        btn.addEventListener('click', () => loadSampleData(i));
        container.appendChild(btn);
    });
}

function loadSampleData(index) {
    const sample = sampleData[index];
    if (!sample) return;

    // Fill age
    const ageInput = document.getElementById('chronological_age');
    ageInput.value = sample.chronological_age;
    ageInput.dispatchEvent(new Event('input', { bubbles: true }));

    // Fill features
    for (const [fname, val] of Object.entries(sample.features)) {
        const input = document.getElementById(fname);
        if (input) {
            input.value = val;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    // Visual feedback
    const btns = document.querySelectorAll('.sample-btn');
    btns.forEach((b, i) => {
        b.style.background = i === index ? 'rgba(99,102,241,0.3)' : '';
        b.style.borderColor = i === index ? 'var(--accent-primary)' : '';
    });
}


// ═══════════════ FORM HANDLING ═══════════════

function setupFormHandlers() {
    const form = document.getElementById('prediction-form');
    const clearBtn = document.getElementById('clear-btn');

    form.addEventListener('submit', handlePredict);
    clearBtn.addEventListener('click', clearForm);
}

function clearForm() {
    document.getElementById('prediction-form').reset();
    document.getElementById('results-section').style.display = 'none';
    document.querySelectorAll('.sample-btn').forEach(b => {
        b.style.background = '';
        b.style.borderColor = '';
    });
}

async function handlePredict(e) {
    e.preventDefault();

    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoading = predictBtn.querySelector('.btn-loading');

    // Validate
    const features = {};
    for (const fname of Object.keys(FEATURE_DEFAULTS)) {
        const input = document.getElementById(fname);
        if (!input.value) {
            input.focus();
            input.style.borderColor = 'var(--red)';
            setTimeout(() => input.style.borderColor = '', 2000);
            return;
        }
        features[fname] = parseFloat(input.value);
    }

    const ageInput = document.getElementById('chronological_age');
    const chronologicalAge = ageInput.value ? parseFloat(ageInput.value) : null;

    // Loading state
    predictBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                features, 
                chronological_age: chronologicalAge,
                model: activeModelKey // Send selected model
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Prediction failed');
        }

        const result = await res.json();
        displayResults(result);

    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        predictBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}


// ═══════════════ DISPLAY RESULTS ═══════════════

function displayResults(result) {
    const section = document.getElementById('results-section');
    section.style.display = 'block';

    // Smooth scroll to results
    setTimeout(() => {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Gauge
    drawGauge(result.predicted_age);
    document.getElementById('gauge-value').textContent = result.predicted_age.toFixed(1);

    // Brain age gap
    displayGap(result);

    // Model info
    document.getElementById('active-model-name').textContent = result.model_name;
    document.getElementById('r2-score').textContent = result.model_r2.toFixed(4);
    document.getElementById('cv-r2-score').textContent = result.model_cv_r2.toFixed(4);
    document.getElementById('mae-score').textContent = result.model_mae.toFixed(2) + 'y';

    // Comparison Table
    renderComparisonTable(result);

    // Assessment
    displayAssessment(result.recommendations.overall_assessment);

    // Charts
    drawSHAPChart(result.feature_contributions, result.base_value, result.predicted_age, result.chronological_age);

    // Recommendations
    displayRecommendations(result.recommendations);
}


function renderComparisonTable(result) {
    const tbody = document.getElementById('comparison-tbody');
    tbody.innerHTML = '';
    
    if (!result.all_model_scores) return;

    const scores = result.all_model_scores;
    const isEnsemble = result.model_key === 'ensemble';
    const activeKey = result.model_key;
    
    const modelKeys = Object.keys(scores);
    
    modelKeys.forEach(key => {
        const s = scores[key];
        const isActive = key === activeKey && !isEnsemble;
        
        let predVal = '--';
        if (isEnsemble && result.individual_predictions) {
            predVal = result.individual_predictions[key].toFixed(1) + ' yrs';
        } else if (key === activeKey) {
            predVal = result.predicted_age.toFixed(1) + ' yrs';
        }

        const tr = document.createElement('tr');
        if (isActive) tr.className = 'active-row';
        
        tr.innerHTML = `
            <td>${s.name} ${isActive ? '✓' : ''}</td>
            <td class="mono">${predVal}</td>
            <td class="mono">${s.r2.toFixed(4)}</td>
            <td class="mono">${s.cv_r2.toFixed(4)}</td>
            <td class="mono ${s.mae < 4.0 ? 'best-score' : ''}">${s.mae.toFixed(2)}y</td>
        `;
        tbody.appendChild(tr);
    });
    
    // Add Ensemble row if active
    if (isEnsemble) {
        const tr = document.createElement('tr');
        tr.className = 'active-row';
        tr.innerHTML = `
            <td><strong>${result.model_name} ✓</strong></td>
            <td class="mono"><strong>${result.predicted_age.toFixed(1)} yrs</strong></td>
            <td class="mono"><strong>${result.model_r2.toFixed(4)}</strong></td>
            <td class="mono"><strong>${result.model_cv_r2.toFixed(4)}</strong></td>
            <td class="mono best-score"><strong>${result.model_mae.toFixed(2)}y</strong></td>
        `;
        tbody.appendChild(tr);
    }
}


function displayGap(result) {
    const gapEl = document.getElementById('gap-value');
    const gapDesc = document.getElementById('gap-description');

    if (result.brain_age_gap !== null) {
        const gap = result.brain_age_gap;
        const sign = gap >= 0 ? '+' : '';
        gapEl.textContent = `${sign}${gap.toFixed(1)} yrs`;

        if (gap <= -2) {
            gapEl.style.color = 'var(--green)';
            gapDesc.textContent = `Your brain appears ${Math.abs(gap).toFixed(1)} years younger than your chronological age of ${result.chronological_age}.`;
        } else if (gap <= 2) {
            gapEl.style.color = 'var(--amber)';
            gapDesc.textContent = `Your brain age is close to your chronological age of ${result.chronological_age}. This is normal.`;
        } else {
            gapEl.style.color = 'var(--red)';
            gapDesc.textContent = `Your brain appears ${gap.toFixed(1)} years older than your chronological age of ${result.chronological_age}.`;
        }
    } else {
        gapEl.textContent = 'N/A';
        gapEl.style.color = 'var(--text-muted)';
        gapDesc.textContent = 'Enter your chronological age above to see the brain age gap.';
    }
}


function displayAssessment(assessment) {
    const card = document.getElementById('assessment-card');
    const icons = {
        excellent: '🌟', good: '✅', normal: '👍', attention: '⚠️', concern: '🔴', info: 'ℹ️'
    };

    document.getElementById('assessment-icon').textContent = icons[assessment.level] || '🧠';
    document.getElementById('assessment-title').textContent = assessment.title;
    document.getElementById('assessment-summary').textContent = assessment.summary;
    card.style.borderColor = assessment.color;
    card.style.boxShadow = `0 0 20px ${assessment.color}22`;
}


// ═══════════════ GAUGE DRAWING ═══════════════

function drawGauge(predictedAge) {
    const canvas = document.getElementById('gauge-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = 280 * dpr;
    canvas.height = 200 * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = '280px';
    canvas.style.height = '200px';

    const cx = 140, cy = 150;
    const radius = 110;
    const startAngle = Math.PI;
    const endAngle = 2 * Math.PI;

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(99,102,241,0.1)';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Age range: 20 to 85
    const minAge = 20, maxAge = 85;
    const clampedAge = Math.max(minAge, Math.min(maxAge, predictedAge));
    const fraction = (clampedAge - minAge) / (maxAge - minAge);
    const ageAngle = startAngle + fraction * Math.PI;

    // Colored arc
    const gradient = ctx.createLinearGradient(30, cy, 250, cy);
    gradient.addColorStop(0, '#22c55e');
    gradient.addColorStop(0.4, '#facc15');
    gradient.addColorStop(0.7, '#f97316');
    gradient.addColorStop(1, '#ef4444');

    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, ageAngle);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Needle dot
    const dotX = cx + radius * Math.cos(ageAngle);
    const dotY = cy + radius * Math.sin(ageAngle);
    ctx.beginPath();
    ctx.arc(dotX, dotY, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(dotX, dotY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = '#6366f1';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('20', 25, cy + 18);
    ctx.fillText('85', 255, cy + 18);
    ctx.fillText('50', cx, 30);
}


// ═══════════════ COMBINED SHAP & LIME CONTRIBUTIONS CHART ═══════════════

function drawSHAPChart(contributions, baseValue, predictedAge, chronologicalAge) {
    const canvas = document.getElementById('shap-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const w = canvas.parentElement.offsetWidth - 48;
    const h = Math.max(450, contributions.length * 45 + 100);

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    ctx.clearRect(0, 0, w, h);

    const isUsingAdjusted = chronologicalAge != null;
    const getShap = c => (isUsingAdjusted && c.adjusted_shap_value !== undefined) ? c.adjusted_shap_value : c.shap_value;
    const getLime = c => c.lime_value || 0;

    // Sort by absolute SHAP value (ascending for bottom-to-top display, wait, I mean descending for top-to-bottom)
    const sorted = [...contributions].sort((a, b) => Math.abs(getShap(b)) - Math.abs(getShap(a)));

    const labelWidth = 200;
    const valueWidth = 60;
    const chartLeft = labelWidth + 10;
    const chartRight = w - valueWidth - 10;
    const chartWidth = chartRight - chartLeft;

    // Find extent of values for scaling based on BOTH shap and lime
    const allVals = [];
    sorted.forEach(c => {
        allVals.push(getShap(c));
        allVals.push(getLime(c));
    });
    
    const minVal = Math.min(...allVals, 0);
    const maxVal = Math.max(...allVals, 0);
    const extent = Math.max(Math.abs(minVal), Math.abs(maxVal)) * 1.3;

    const rowHeight = 36;
    const barHeight = 12;
    const barGap = 2;
    const topPad = 60;

    // Title & Legend Area
    ctx.textAlign = 'left';
    
    // Title
    ctx.fillStyle = '#94a3b8';
    ctx.font = '13px Inter, sans-serif';
    if (isUsingAdjusted) {
        ctx.fillText(`Age Baseline: ${chronologicalAge.toFixed(1)}y → Predicted: ${predictedAge.toFixed(1)}y`, chartLeft, 22);
    } else {
        ctx.fillText(`Population Baseline: ${baseValue.toFixed(1)}y → Predicted: ${predictedAge.toFixed(1)}y`, chartLeft, 22);
    }

    // Legend
    ctx.font = '11px Inter, sans-serif';
    
    // SHAP Aging
    ctx.fillStyle = 'rgba(239, 68, 68, 0.9)'; // Red
    ctx.beginPath(); ctx.roundRect(chartRight - 120, 10, 10, 10, 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0'; ctx.fillText('SHAP (Older)', chartRight - 105, 19);
    
    // SHAP Youthful
    ctx.fillStyle = 'rgba(34, 197, 94, 0.9)'; // Green
    ctx.beginPath(); ctx.roundRect(chartRight - 120, 26, 10, 10, 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0'; ctx.fillText('SHAP (Younger)', chartRight - 105, 35);
    
    // LIME Aging
    ctx.fillStyle = 'rgba(245, 158, 11, 0.9)'; // Orange
    ctx.beginPath(); ctx.roundRect(chartRight - 230, 10, 10, 10, 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0'; ctx.fillText('LIME (Older)', chartRight - 215, 19);
    
    // LIME Youthful
    ctx.fillStyle = 'rgba(14, 165, 233, 0.9)'; // Sky Blue
    ctx.beginPath(); ctx.roundRect(chartRight - 230, 26, 10, 10, 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0'; ctx.fillText('LIME (Younger)', chartRight - 215, 35);

    // Zero line
    const zeroX = chartLeft + (chartWidth / 2);
    ctx.beginPath();
    ctx.moveTo(zeroX, topPad - 10);
    ctx.lineTo(zeroX, h - 30);
    ctx.strokeStyle = 'rgba(148,163,184,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw bars
    sorted.forEach((c, i) => {
        const rowY = topPad + i * rowHeight;
        
        const shapVal = getShap(c);
        const limeVal = getLime(c);
        
        const shapW = (Math.abs(shapVal) / extent) * (chartWidth / 2);
        const limeW = (Math.abs(limeVal) / extent) * (chartWidth / 2);

        // Label
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(c.display_name, labelWidth, rowY + rowHeight / 2 + 4);

        // ── SHAP Bar (Top) ──
        const isShapPositive = shapVal >= 0;
        const shapX = isShapPositive ? zeroX : zeroX - shapW;
        const shapY = rowY + 4;
        
        ctx.beginPath();
        ctx.roundRect(shapX, shapY, shapW, barHeight, 3);
        ctx.fillStyle = isShapPositive ? 'rgba(239, 68, 68, 0.85)' : 'rgba(34, 197, 94, 0.85)';
        ctx.fill();
        
        // SHAP Value Text
        ctx.fillStyle = isShapPositive ? '#fca5a5' : '#86efac';
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.textAlign = isShapPositive ? 'left' : 'right';
        ctx.fillText(
            `${isShapPositive ? '+' : ''}${shapVal.toFixed(2)}`,
            isShapPositive ? shapX + shapW + 6 : shapX - 6,
            shapY + barHeight - 2
        );

        // ── LIME Bar (Bottom) ──
        const isLimePositive = limeVal >= 0;
        const limeX = isLimePositive ? zeroX : zeroX - limeW;
        const limeY = shapY + barHeight + barGap;
        
        ctx.beginPath();
        ctx.roundRect(limeX, limeY, limeW, barHeight, 3);
        ctx.fillStyle = isLimePositive ? 'rgba(245, 158, 11, 0.85)' : 'rgba(14, 165, 233, 0.85)';
        ctx.fill();
        
        // LIME Value Text
        ctx.fillStyle = isLimePositive ? '#fcd34d' : '#7dd3fc';
        ctx.textAlign = isLimePositive ? 'left' : 'right';
        ctx.fillText(
            `${isLimePositive ? '+' : ''}${limeVal.toFixed(2)}`,
            isLimePositive ? limeX + limeW + 6 : limeX - 6,
            limeY + barHeight - 2
        );
    });

    // Bottom Axis Labels
    const legendY = h - 10;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#86efac';
    ctx.fillText('← Younger brain', chartLeft + chartWidth * 0.25, legendY);
    ctx.fillStyle = '#fca5a5';
    ctx.fillText('Older brain →', chartLeft + chartWidth * 0.75, legendY);
}



// ═══════════════ RECOMMENDATIONS DISPLAY ═══════════════

function displayRecommendations(recommendations) {
    // Feature-specific
    const grid = document.getElementById('recommendations-grid');
    grid.innerHTML = '';

    const featureRecs = recommendations.feature_recommendations || [];

    // Show top impactful ones (attention first, then positive)
    const attention = featureRecs.filter(r => r.status === 'attention');
    const positive = featureRecs.filter(r => r.status === 'positive');
    const neutral = featureRecs.filter(r => r.status === 'neutral');
    const ordered = [...attention, ...positive, ...neutral];

    ordered.forEach(rec => {
        const card = document.createElement('div');
        card.className = `rec-card status-${rec.status}`;
        card.innerHTML = `
            <div class="rec-header">
                <span class="rec-icon">${rec.icon}</span>
                <span class="rec-region">${rec.region}</span>
                <span class="rec-status">${rec.status_label}</span>
            </div>
            <div class="rec-value" style="line-height:1.4;">
                Value: ${rec.value}${rec.unit ? ' ' + rec.unit : ''} · Raw Impact: ${rec.shap_value >= 0 ? '+' : ''}${rec.shap_value.toFixed(2)}y<br>
                <span style="opacity:0.8;font-size:0.9em">Age-Adjusted Impact: ${rec.adjusted_shap_value !== undefined ? (rec.adjusted_shap_value >= 0 ? '+' : '') + rec.adjusted_shap_value.toFixed(2) + 'y' : 'N/A'}</span>
            </div>
            <ul class="rec-tips">
                ${rec.tips.map(t => `<li>${t}</li>`).join('')}
            </ul>
        `;
        grid.appendChild(card);
    });

    // General tips
    const tipsContainer = document.getElementById('general-tips');
    tipsContainer.innerHTML = '';

    const generalTips = recommendations.general_tips || [];
    generalTips.forEach(tip => {
        const card = document.createElement('div');
        card.className = 'tip-card';
        card.innerHTML = `
            <span class="tip-icon">${tip.icon}</span>
            <div>
                <div class="tip-title">${tip.title}</div>
                <div class="tip-detail">${tip.detail}</div>
            </div>
        `;
        tipsContainer.appendChild(card);
    });
}
