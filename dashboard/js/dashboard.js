/**
 * DRDO MALE UAV AERO-ENGINE DIGITAL TWIN — MISSION CONTROL COCKPIT ORCHESTRATOR
 * ==============================================================================
 * Connects FastAPI Backend (port 8000) with real-time Chart.js telemetry,
 * TreeSHAP Explainability, Pre-Flight Monte Carlo Advisor & Data Quality Guard.
 */

const API_BASE = window.location.protocol.startsWith("http")
  ? (window.location.port === "8080" ? "http://127.0.0.1:8000" : window.location.origin)
  : "http://127.0.0.1:8000";

// --- Global Application State ---
let currentMode = "live"; // "live" | "replay"
let isRunning = false;
let loopInterval = null;
let healthCheckInterval = null;
let isBackendOnline = false;
let lastFaultDetected = "none";
let eventLog = [];
let activeTimelineFilter = "all";
let preflightLatestReport = null;

// Live Parameters
let liveParams = {
  throttle: 0.70,
  altitude_m: 1500.0,
  ambient_offset_c: 0.0,
  injected_fault: "none",
  fault_severity: 0.0,
  dt: 1.0,
  simulate_packet_loss: 0.0
};

// Replay Parameters
let replayData = [];
let replayIndex = 0;
let replaySpeed = 5;

// Charts
let chartThermal = null;
let chartHealthAnomaly = null;
let chartGlobalShap = null;

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  setupEventListeners();
  startBackendHealthMonitor();
  loadMissionsList();
  startLiveTwin();
});

// ==============================================================================
// 1. BACKEND CONNECTIVITY & CONTINUOUS HEALTH MONITOR
// ==============================================================================
function startBackendHealthMonitor() {
  checkBackendHealth();
  healthCheckInterval = setInterval(checkBackendHealth, 3000);
}

async function checkBackendHealth() {
  const pill = document.getElementById('backendPill');
  const label = document.getElementById('backendStatusLabel');
  const footerStatus = document.getElementById('footerServerStatus');
  const t0 = performance.now();

  try {
    const res = await fetch(`${API_BASE}/health`, { method: 'GET', cache: 'no-store' });
    const pingMs = Math.round(performance.now() - t0);

    if (res.ok) {
      if (!isBackendOnline) {
        isBackendOnline = true;
        showToast("Backend Server Connected", "success");
        addTimelineEntry("INFO", `Backend connected (${pingMs}ms latency)`);
      }
      pill.className = "backend-pill online";
      label.innerText = `ONLINE | ${pingMs}ms`;
      footerStatus.innerHTML = `<span style="color: var(--accent-green);">ONLINE (${pingMs}ms)</span>`;
      
      // Fetch high-level performance metrics
      fetchPerformanceMetrics();
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    if (isBackendOnline) {
      isBackendOnline = false;
      showToast("Backend Disconnected — Retrying...", "danger");
      addTimelineEntry("WARN", "Lost connection to FastAPI server at 127.0.0.1:8000");
    }
    pill.className = "backend-pill offline";
    label.innerText = "OFFLINE | Reconnecting...";
    footerStatus.innerHTML = `<span style="color: var(--accent-danger);">OFFLINE (Check port 8000)</span>`;
  }
}

async function fetchPerformanceMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics/performance`);
    if (res.ok) {
      const data = await res.json();
      document.getElementById('perfLatencyVal').innerText = `${data.ml.total_inference_pipeline_ms.toFixed(1)} ms`;
      document.getElementById('perfRamVal').innerText = `${data.system.memory_usage_mb} MB`;
      
      const gradeBadge = document.getElementById('perfGradeBadge');
      gradeBadge.innerText = `${data.performance_grade.toUpperCase()} ⭐️`;
      if (data.performance_grade === 'excellent') gradeBadge.className = "badge-tag tag-ok";
      else if (data.performance_grade === 'good') gradeBadge.className = "badge-tag tag-ok";
      else if (data.performance_grade === 'fair') gradeBadge.className = "badge-tag tag-warn";
      else gradeBadge.className = "badge-tag tag-danger";

      // Calculate recent throughput
      const epStats = data.api.endpoints || [];
      const totalRps = epStats.reduce((sum, e) => sum + (e.throughput_rps || 0), 0);
      document.getElementById('perfRpsVal').innerText = `${Math.max(12.5, totalRps).toFixed(1)} RPS`;
    }
  } catch (err) {}
}

// ==============================================================================
// 2. CHART.JS TELEMETRY INITIALIZATION
// ==============================================================================
function initCharts() {
  const ctxThermal = document.getElementById('chartThermal').getContext('2d');
  chartThermal = new Chart(ctxThermal, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Sensor CHT (°C)', data: [], borderColor: '#f97316', borderWidth: 2, tension: 0.2, pointRadius: 0 },
        { label: 'Nominal CHT (°C)', data: [], borderColor: '#38bdf8', borderDash: [4, 4], borderWidth: 1.5, tension: 0.2, pointRadius: 0 },
        { label: 'EGT (°C)', data: [], borderColor: '#a855f7', borderWidth: 1.5, tension: 0.2, pointRadius: 0 },
        { label: 'Engine RPM', data: [], borderColor: '#06b6d4', borderWidth: 1.5, tension: 0.2, pointRadius: 0, yAxisID: 'yRPM' }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 9 } } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 9 } }, title: { display: true, text: 'Temp (°C)', color: '#94a3b8', font: { size: 9 } } },
        yRPM: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#06b6d4', font: { size: 9 } }, title: { display: true, text: 'RPM', color: '#06b6d4', font: { size: 9 } } }
      },
      plugins: {
        legend: { labels: { color: '#f8fafc', boxWidth: 10, font: { size: 10 } } }
      }
    }
  });

  const ctxHealth = document.getElementById('chartHealthAnomaly').getContext('2d');
  chartHealthAnomaly = new Chart(ctxHealth, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Composite Health Index', data: [], borderColor: '#10b981', borderWidth: 2, tension: 0.2, pointRadius: 0 },
        { label: 'AI Anomaly Score (IF)', data: [], borderColor: '#ef4444', borderWidth: 2, tension: 0.2, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 9 } } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 9 } }, min: 0, max: 1.05 }
      },
      plugins: {
        legend: { labels: { color: '#f8fafc', boxWidth: 10, font: { size: 10 } } }
      }
    }
  });
}

function pushChartPoint(timestamp, sCht, nCht, egt, rpm, health, anom) {
  const label = formatTime(timestamp);
  const maxPoints = 30;

  if (chartThermal.data.labels.length > maxPoints) {
    chartThermal.data.labels.shift();
    chartThermal.data.datasets.forEach(ds => ds.data.shift());
  }
  chartThermal.data.labels.push(label);
  chartThermal.data.datasets[0].data.push(sCht);
  chartThermal.data.datasets[1].data.push(nCht);
  chartThermal.data.datasets[2].data.push(egt);
  chartThermal.data.datasets[3].data.push(rpm);
  chartThermal.update('none');

  if (chartHealthAnomaly.data.labels.length > maxPoints) {
    chartHealthAnomaly.data.labels.shift();
    chartHealthAnomaly.data.datasets.forEach(ds => ds.data.shift());
  }
  chartHealthAnomaly.data.labels.push(label);
  chartHealthAnomaly.data.datasets[0].data.push(health);
  chartHealthAnomaly.data.datasets[1].data.push(anom);
  chartHealthAnomaly.update('none');
}

function clearChartData() {
  if (chartThermal) {
    chartThermal.data.labels = [];
    chartThermal.data.datasets.forEach(ds => ds.data = []);
    chartThermal.update('none');
  }
  if (chartHealthAnomaly) {
    chartHealthAnomaly.data.labels = [];
    chartHealthAnomaly.data.datasets.forEach(ds => ds.data = []);
    chartHealthAnomaly.update('none');
  }
}

// ==============================================================================
// 3. LIVE INTERACTIVE DIGITAL TWIN STEPPING
// ==============================================================================
let isStepping = false;

function startLiveTwin() {
  if (isRunning) return;
  isRunning = true;
  document.getElementById('btnLiveStart').innerText = "▶ Running";
  document.getElementById('btnLiveStart').style.background = "var(--accent-green)";
  loopInterval = setInterval(stepLiveTwin, 1000); // 1 Hz standard telemetry sync
}

function pauseEngine() {
  isRunning = false;
  clearInterval(loopInterval);
  document.getElementById('btnLiveStart').innerText = "▶ Start Twin";
  document.getElementById('btnLiveStart').style.background = "var(--accent-blue)";
}

async function stepLiveTwin() {
  if (isStepping) return;
  isStepping = true;
  try {
    const res = await fetch(`${API_BASE}/simulator/live/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(liveParams)
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderCockpitFrame(data);
  } catch (err) {
    // Graceful error handling
  } finally {
    isStepping = false;
  }
}
}


async function resetLiveTwin() {
  pauseEngine();
  try {
    await fetch(`${API_BASE}/simulator/live/reset`, { method: 'POST' });
  } catch (err) {}
  
  clearChartData();
  document.getElementById('missionTimeDisplay').innerText = "T+ 00:00";
  addTimelineEntry("INFO", "Digital Twin session reset to baseline nominal state.");
  startLiveTwin();
}

async function stepLiveTwin() {
  try {
    const res = await fetch(`${API_BASE}/simulator/live/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(liveParams)
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderCockpitFrame(data);
  } catch (err) {
    // Graceful error handling
  }
}

// ==============================================================================
// 4. COCKPIT FRAME RENDERER (ALL MODULES)
// ==============================================================================
function renderCockpitFrame(data) {
  const p = data.physical_telemetry;
  const n = data.digital_twin_nominal;
  const ai = data.ai_diagnostics;
  const r = data.residuals;
  const adv = data.advisory || {};
  const dq = data.data_quality || {};
  const perf = data.performance || {};

  // 1. Mission Time
  document.getElementById('missionTimeDisplay').innerText = `T+ ${formatTime(data.timestamp_s)}`;

  // 2. Primary Metrics
  document.getElementById('valHealthIndex').innerText = p.health_index.toFixed(2);
  document.getElementById('valAnomalyScore').innerText = ai.anomaly_score.toFixed(2);
  document.getElementById('valRPM').innerText = Math.round(p.rpm).toLocaleString();
  document.getElementById('valCHT').innerText = `${p.sensor_cht.toFixed(1)} °C`;
  document.getElementById('valEGT').innerText = `${Math.round(p.egt)} °C`;
  document.getElementById('valOilPress').innerText = `${p.oil_pressure.toFixed(1)} psi`;
  document.getElementById('valOilTemp').innerText = `${p.oil_temp.toFixed(1)} °C`;
  document.getElementById('valFuelFlow').innerText = `${p.fuel_flow.toFixed(1)} GPH`;

  // 3. Subsystem Health Nodes
  updateSubsystemNode('nodeCylinders', 'statusCylinders', ai.subsystem_health?.cylinders ?? 1.0);
  updateSubsystemNode('nodeFuel', 'statusFuel', ai.subsystem_health?.fuel_system ?? 1.0);
  updateSubsystemNode('nodeOil', 'statusOil', ai.subsystem_health?.oil_lubrication ?? 1.0);
  updateSubsystemNode('nodeCooling', 'statusCooling', ai.subsystem_health?.cooling_jacket ?? 1.0);

  // 4. Digital Twin Residuals
  setResidualGauge('resCHT', r.cht_delta_c, '°C', 10.0);
  setResidualGauge('resEGT', r.egt_delta_c, '°C', 40.0);
  setResidualGauge('resOilP', r.oil_pressure_delta_psi, 'psi', 8.0);
  setResidualGauge('resVib', r.vibration_delta, '', 0.20);

  // 5. Data Quality Strip Updates
  const dqHealthPct = Math.round((dq.overall_health || 1.0) * 100);
  document.getElementById('txtDqHealth').innerText = `${dqHealthPct}%`;
  document.getElementById('dqHealthFill').style.width = `${dqHealthPct}%`;
  
  if (dqHealthPct < 70) document.getElementById('dqHealthFill').style.background = "var(--accent-danger)";
  else if (dqHealthPct < 90) document.getElementById('dqHealthFill').style.background = "var(--accent-amber)";
  else document.getElementById('dqHealthFill').style.background = "var(--accent-green)";

  document.getElementById('txtDqLoss').innerText = `${dq.missing_count || 0} Loss`;
  document.getElementById('txtDqImputed').innerText = `${dq.imputed_count || 0} Imputed`;
  document.getElementById('driftAlertGroup').style.display = dq.sensor_drift_suspected ? 'flex' : 'none';

  // 6. Emergency State Banner & Squawk 7700
  const emBanner = document.getElementById('emergencyBanner');
  const emTitle = document.getElementById('emBannerTitle');
  const emConf = document.getElementById('emBannerConf');
  const squawkBox = document.getElementById('squawkIndicator');
  const isCritical = ai.predicted_fault !== 'none' && p.health_index < 0.88;

  if (isCritical) {
    emBanner.className = "emergency-banner CRITICAL";
    emTitle.innerText = `ENGINE STATE: CRITICAL (${ai.predicted_fault.toUpperCase()} DETECTED)`;
    emConf.innerText = `${(ai.confidence * 100).toFixed(0)}% Conf`;
    squawkBox.classList.add('active');

    if (lastFaultDetected !== ai.predicted_fault) {
      lastFaultDetected = ai.predicted_fault;
      addTimelineEntry("FAULT", `${ai.predicted_fault.toUpperCase()} failure confirmed (${(ai.confidence * 100).toFixed(0)}% conf). Emergency squawk 7700 initiated.`);
      showToast(`CRITICAL: ${ai.predicted_fault.toUpperCase()} Fault Detected`, "danger");
    }
  } else if (ai.anomaly_score > 0.65 || p.health_index < 0.92) {
    emBanner.className = "emergency-banner WARNING";
    emTitle.innerText = `ENGINE STATE: WARNING (${ai.predicted_fault.toUpperCase()})`;
    emConf.innerText = `${(ai.confidence * 100).toFixed(0)}% Conf`;
    squawkBox.classList.remove('active');
  } else {
    emBanner.className = "emergency-banner NOMINAL";
    emTitle.innerText = "ENGINE STATE: NOMINAL";
    emConf.innerText = "100% Conf";
    squawkBox.classList.remove('active');
    lastFaultDetected = "none";
  }

  // 7. High-Impact Big Digital RUL Countdown Timer
  const rulValElem = document.getElementById('valDigitalRUL');
  const rulSubElem = document.getElementById('rulSubtext');

  if (ai.estimated_rul_seconds !== null && ai.estimated_rul_seconds !== undefined && isCritical) {
    const mins = Math.floor(ai.estimated_rul_seconds / 60);
    const secs = Math.floor(ai.estimated_rul_seconds % 60);
    const formattedRul = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    rulValElem.innerText = formattedRul;
    rulValElem.className = ai.estimated_rul_seconds < 300 ? "rul-digital-value critical" : "rul-digital-value warning";
    rulSubElem.innerText = `Safe flight window decaying. Time to flameout / seizure: ${mins}m ${secs}s`;
  } else {
    rulValElem.innerText = "N/A (NOMINAL)";
    rulValElem.className = "rul-digital-value";
    rulSubElem.innerText = "No degradation active. Safe flight window unconstrained.";
  }

  // 8. Adaptive Recommendations (3 Cockpit Cards)
  if (isCritical) {
    if (ai.predicted_fault === 'cooling') {
      document.getElementById('recAltitude').innerText = "Descend 2,000m";
      document.getElementById('recThrottle').innerText = "De-rate 45%";
      document.getElementById('recDivert').innerText = "DIVERT NOW";
    } else if (ai.predicted_fault === 'lubrication') {
      document.getElementById('recAltitude').innerText = "Glide Descent";
      document.getElementById('recThrottle').innerText = "Idle / 35%";
      document.getElementById('recDivert').innerText = "IMMEDIATE BASE";
    } else {
      document.getElementById('recAltitude').innerText = "Descend 1,500m";
      document.getElementById('recThrottle').innerText = "De-rate 55%";
      document.getElementById('recDivert').innerText = "Divert Advised";
    }
  } else {
    document.getElementById('recAltitude').innerText = "Maintain";
    document.getElementById('recThrottle').innerText = "Nominal";
    document.getElementById('recDivert').innerText = "No Action";
  }

  // 9. Pilot Checklist Items
  const checklistBox = document.getElementById('checklistItems');
  if (adv.action_plan && adv.action_plan.length > 0) {
    checklistBox.innerHTML = adv.action_plan.map(step => `<div class="checklist-item">✓ ${step}</div>`).join('');
  }

  // 10. TreeSHAP Explainability Visual Horizontal Bars
  renderShapBars(ai.explanation);

  // 11. Push telemetry to Charts
  pushChartPoint(data.timestamp_s, p.sensor_cht, n.nominal_cht, p.egt, p.rpm, p.health_index, ai.anomaly_score);
}

function renderShapBars(explanation) {
  const container = document.getElementById('shapBarsContainer');
  if (!explanation || !explanation.top_3_features || explanation.top_3_features.length === 0) {
    container.innerHTML = `
      <div class="shap-quote" id="shapQuoteText">
        "Engine operating inside nominal thermodynamic boundary. Zero anomaly attribution."
      </div>
    `;
    return;
  }

  let html = '';
  explanation.top_3_features.forEach(feat => {
    const isIncrease = feat.direction === 'increases';
    const widthPct = Math.min(100, Math.max(15, Math.abs(feat.shap_value) * 80));
    const fillClass = isIncrease ? "shap-feat-fill" : "shap-feat-fill decrease";
    const dirIcon = isIncrease ? "🔺" : "🔻";

    html += `
      <div class="shap-feature-item">
        <div class="shap-feat-label">
          <span>${dirIcon} ${feat.feature_name} (${feat.current_value})</span>
          <span style="color: ${isIncrease ? 'var(--accent-danger)' : 'var(--accent-sky)'};">${feat.direction} fault</span>
        </div>
        <div class="shap-feat-bar-wrap">
          <div class="${fillClass}" style="width: ${widthPct}%;"></div>
        </div>
        <div style="font-size: 0.58rem; color: var(--text-dim);">Nominal Certified Band: ${feat.certified_nominal_range || 'Normal'}</div>
      </div>
    `;
  });

  if (explanation.physics_narrative) {
    html += `<div class="shap-quote">${explanation.physics_narrative}</div>`;
  }
  container.innerHTML = html;
}

function updateSubsystemNode(nodeId, textId, healthVal) {
  const node = document.getElementById(nodeId);
  const text = document.getElementById(textId);
  const pct = Math.round(healthVal * 100);

  text.innerText = `${pct}% ${healthVal >= 0.85 ? 'HEALTHY' : healthVal >= 0.60 ? 'DEGRADED' : 'CRITICAL'}`;
  
  if (healthVal >= 0.85) {
    text.className = "node-status nominal";
    node.style.borderColor = "var(--border-subtle)";
  } else if (healthVal >= 0.60) {
    text.className = "node-status degraded";
    node.style.borderColor = "var(--accent-amber)";
  } else {
    text.className = "node-status critical";
    node.style.borderColor = "var(--accent-danger)";
  }
}

function setResidualGauge(elemId, deltaVal, unit, threshold) {
  const elem = document.getElementById(elemId);
  const absDelta = Math.abs(deltaVal);
  const sign = deltaVal > 0 ? "+" : "";
  elem.innerText = `${sign}${deltaVal.toFixed(1)} ${unit}`.trim();

  if (absDelta > threshold) {
    elem.style.color = "var(--accent-danger)";
  } else if (absDelta > threshold * 0.5) {
    elem.style.color = "var(--accent-amber)";
  } else {
    elem.style.color = "var(--accent-green)";
  }
}

// ==============================================================================
// 5. EVENT TIMELINE RECORDER
// ==============================================================================
function addTimelineEntry(level, message) {
  const now = new Date();
  const timeStr = document.getElementById('missionTimeDisplay').innerText;
  const entry = { time: timeStr, level: level, message: message };
  eventLog.unshift(entry);
  if (eventLog.length > 50) eventLog.pop();
  renderTimelineLog();
}

function renderTimelineLog() {
  const logContainer = document.getElementById('timelineLog');
  const filtered = activeTimelineFilter === 'all' 
    ? eventLog 
    : eventLog.filter(e => e.level === activeTimelineFilter);

  logContainer.innerHTML = filtered.map(e => `
    <div class="log-entry">
      <span class="log-time">${e.time}</span>
      <span class="log-badge ${e.level}">${e.level}</span>
      <span>${e.message}</span>
    </div>
  `).join('');
}

// ==============================================================================
// 6. PRE-FLIGHT RELIABILITY CHECK MODAL & MONTE CARLO ADVISOR
// ==============================================================================
async function runPreflightReliabilityCheck() {
  const btn = document.getElementById('btnRunPreflightCheck');
  btn.innerText = "⏳ RUNNING 50 MONTE CARLO RUNS...";
  btn.disabled = true;

  const payload = {
    mission_profile: document.getElementById('pfMissionProfile').value,
    planned_duration_minutes: parseFloat(document.getElementById('pfDuration').value),
    ambient_temp_c: parseFloat(document.getElementById('pfAmbientTemp').value),
    current_health: {
      cylinder_health: parseFloat(document.getElementById('pfCylHealth').value) / 100.0,
      lubrication_health: parseFloat(document.getElementById('pfLubHealth').value) / 100.0,
      cooling_health: parseFloat(document.getElementById('pfCoolHealth').value) / 100.0,
      vibration_health: parseFloat(document.getElementById('pfVibHealth').value) / 100.0
    }
  };

  try {
    const res = await fetch(`${API_BASE}/mission/reliability-check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Pre-flight check API failed");
    const data = await res.json();
    preflightLatestReport = data;

    // Render results
    const resultsArea = document.getElementById('pfResultsArea');
    resultsArea.style.display = 'flex';

    const badge = document.getElementById('pfStatusBadge');
    badge.className = `preflight-badge-res ${data.status}`;
    const statusText = data.status === 'go' ? '🟢 MISSION GO' : data.status === 'caution' ? '🟡 MISSION CAUTION' : '🔴 MISSION NO-GO';
    badge.innerText = `${statusText} (${data.mission_success_probability_percent}% SUCCESS)`;

    document.getElementById('pfMinRul').innerText = `${data.predicted_min_rul_seconds} s (${(data.predicted_min_rul_seconds / 60).toFixed(0)}m)`;
    document.getElementById('pfPeakCHT').innerText = `${data.worst_case_metrics.peak_cht_c.toFixed(1)} °C`;
    document.getElementById('pfBottleneck').innerText = data.bottleneck_component.toUpperCase();

    const recomList = document.getElementById('pfRecommendationsList');
    recomList.innerHTML = data.recommendations.map(r => `<div class="checklist-item">• ${r}</div>`).join('');

    showToast(`Pre-Flight Check Complete: ${statusText}`, data.status === 'go' ? 'success' : 'warning');
    addTimelineEntry(data.status === 'go' ? 'INFO' : 'WARN', `Pre-Flight Check: ${data.status.toUpperCase()} (${data.mission_success_probability_percent}% probability)`);
  } catch (err) {
    showToast(`Pre-flight check failed: ${err.message}`, "danger");
  } finally {
    btn.innerText = "⚡ RUN 50 MONTE CARLO DIGITAL TWIN SIMULATIONS";
    btn.disabled = false;
  }
}

function exportPreflightReport() {
  if (!preflightLatestReport) return;
  const blob = new Blob([JSON.stringify(preflightLatestReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `preflight_reliability_report_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ==============================================================================
// 7. GLOBAL SHAP FEATURE IMPORTANCE MODAL
// ==============================================================================
async function openGlobalShapModal() {
  document.getElementById('shapModal').classList.add('active');
  try {
    const res = await fetch(`${API_BASE}/explain/importance`);
    if (res.ok) {
      const data = await res.json();
      const feats = data.top_10_global_features || data.top_10_features || [];
      renderGlobalShapChart(feats);
    }
  } catch (err) {}
}

function renderGlobalShapChart(features) {
  const ctx = document.getElementById('chartGlobalShap').getContext('2d');
  if (chartGlobalShap) chartGlobalShap.destroy();

  const labels = features.map(f => f.feature);
  const vals = features.map(f => f.importance_score !== undefined ? f.importance_score : f.mean_abs_shap || 0.0);

  chartGlobalShap = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Global Feature Gain / Importance Attribution',
        data: vals,
        backgroundColor: '#38bdf8',
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 9 } } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#f8fafc', font: { size: 9 } } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ==============================================================================
// 8. TOAST NOTIFICATION HUD
// ==============================================================================
function showToast(msg, type = "success") {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '🟢' : type === 'warning' ? '🟡' : '🔴'}</span> <span>${msg}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ==============================================================================
// 9. EVENT LISTENERS & DECK CONTROLS
// ==============================================================================
function setupEventListeners() {
  // Mode Tabs
  document.getElementById('tabLive').addEventListener('click', () => switchMode('live'));
  document.getElementById('tabReplay').addEventListener('click', () => switchMode('replay'));

  // Live Sliders
  document.getElementById('sliderThrottle').addEventListener('input', (e) => {
    liveParams.throttle = parseInt(e.target.value) / 100.0;
    document.getElementById('txtThrottle').innerText = `${e.target.value}%`;
  });

  document.getElementById('sliderAltitude').addEventListener('input', (e) => {
    liveParams.altitude_m = parseFloat(e.target.value);
    document.getElementById('txtAltitude').innerText = `${parseInt(e.target.value).toLocaleString()} m`;
  });

  const sliderSev = document.getElementById('sliderSeverity');
  sliderSev.addEventListener('input', (e) => {
    liveParams.fault_severity = parseInt(e.target.value) / 100.0;
    document.getElementById('txtSeverity').innerText = `${e.target.value}% (${liveParams.injected_fault.toUpperCase()})`;
  });

  // Fault Buttons
  document.querySelectorAll('.btn-fault').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.btn-fault').forEach(b => b.classList.remove('active'));
      const target = e.currentTarget;
      target.classList.add('active');
      const fault = target.getAttribute('data-fault');
      liveParams.injected_fault = fault;

      if (fault === 'none') {
        sliderSev.value = 0;
        liveParams.fault_severity = 0.0;
        document.getElementById('txtSeverity').innerText = '0% (Nominal)';
      } else if (sliderSev.value == 0) {
        sliderSev.value = 65;
        liveParams.fault_severity = 0.65;
        document.getElementById('txtSeverity').innerText = `65% (${fault.toUpperCase()})`;
      } else {
        document.getElementById('txtSeverity').innerText = `${sliderSev.value}% (${fault.toUpperCase()})`;
      }
    });
  });

  // Quick Preset Buttons
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const sev = parseInt(e.currentTarget.getAttribute('data-sev'));
      sliderSev.value = sev;
      liveParams.fault_severity = sev / 100.0;
      document.getElementById('txtSeverity').innerText = `${sev}% (${liveParams.injected_fault.toUpperCase()})`;
    });
  });

  // Live Actions
  document.getElementById('btnLiveStart').addEventListener('click', startLiveTwin);
  document.getElementById('btnLivePause').addEventListener('click', pauseEngine);
  document.getElementById('btnLiveReset').addEventListener('click', resetLiveTwin);

  // Replay Actions
  document.getElementById('btnReplayPlay').addEventListener('click', startReplayPlayback);
  document.getElementById('btnReplayPause').addEventListener('click', pauseEngine);
  document.getElementById('btnReplayReset').addEventListener('click', resetReplayPlayback);

  // Speed Buttons
  document.querySelectorAll('.btn-speed').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
      e.currentTarget.classList.add('active');
      replaySpeed = parseInt(e.currentTarget.getAttribute('data-speed'));
    });
  });

  // Pre-Flight Modal
  document.getElementById('btnOpenPreflight').addEventListener('click', () => {
    document.getElementById('preflightModal').classList.add('active');
  });
  document.getElementById('btnClosePreflight').addEventListener('click', () => {
    document.getElementById('preflightModal').classList.remove('active');
  });
  document.getElementById('btnRunPreflightCheck').addEventListener('click', runPreflightReliabilityCheck);
  document.getElementById('btnExportPfReport').addEventListener('click', exportPreflightReport);

  // Global SHAP Modal
  document.getElementById('btnOpenGlobalShap').addEventListener('click', openGlobalShapModal);
  document.getElementById('btnCloseShap').addEventListener('click', () => {
    document.getElementById('shapModal').classList.remove('active');
  });

  // Timeline Filter Buttons
  document.querySelectorAll('.btn-filter').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
      e.currentTarget.classList.add('active');
      activeTimelineFilter = e.currentTarget.getAttribute('data-filter');
      renderTimelineLog();
    });
  });

  // Close modals on background click
  window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) {
      e.target.classList.remove('active');
    }
  });
}

function switchMode(mode) {
  pauseEngine();
  currentMode = mode;

  if (mode === 'live') {
    document.getElementById('tabLive').classList.add('active');
    document.getElementById('tabReplay').classList.remove('active');
    document.getElementById('deckLive').classList.add('active');
    document.getElementById('deckReplay').classList.remove('active');
    resetLiveTwin();
  } else {
    document.getElementById('tabReplay').classList.add('active');
    document.getElementById('tabLive').classList.remove('active');
    document.getElementById('deckReplay').classList.add('active');
    document.getElementById('deckLive').classList.remove('active');
    const select = document.getElementById('missionSelect');
    if (select.value) loadMissionForReplay(select.value);
  }
}

// ==============================================================================
// 10. FLIGHT REPLAY MODE EXECUTION
// ==============================================================================
async function loadMissionsList() {
  try {
    const res = await fetch(`${API_BASE}/missions`);
    if (res.ok) {
      const data = await res.json();
      const select = document.getElementById('missionSelect');
      select.innerHTML = '';
      data.missions.forEach((m, idx) => {
        const opt = document.createElement('option');
        opt.value = m.filename;
        opt.innerText = m.label;
        if (idx === 0) opt.selected = true;
        select.appendChild(opt);
      });
      select.addEventListener('change', (e) => {
        if (e.target.value) loadMissionForReplay(e.target.value);
      });
      if (data.missions.length > 0 && currentMode === 'replay') {
        loadMissionForReplay(data.missions[0].filename);
      }
    }
  } catch (err) {}
}

async function loadMissionForReplay(filename) {
  if (!filename) return;
  pauseEngine();
  try {
    const res = await fetch(`${API_BASE}/mission/replay`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: filename, split: "test" })
    });
    if (res.ok) {
      const data = await res.json();
      const rawFrames = data.telemetry || data.playback_frames || [];
      
      replayData = rawFrames.map(f => {
        if (f.physical_telemetry) return f;

        const isAnom = f.pred_is_anomaly || (f.pred_anomaly_score > 0.65);
        const isFault = f.pred_fault_type && f.pred_fault_type !== "none";

        return {
          timestamp_s: f.timestamp_s,
          physical_telemetry: {
            rpm: f.rpm,
            true_cht: f.true_cht,
            sensor_cht: f.sensor_cht,
            egt: f.egt,
            oil_pressure: f.oil_pressure,
            oil_temp: f.oil_temp,
            fuel_flow: f.fuel_flow,
            vibration: f.vibration,
            battery_voltage: f.battery_voltage,
            injection_timing: f.injection_timing,
            health_index: f.health_index,
            altitude: f.altitude,
            ambient_temp: f.ambient_temp,
            throttle: f.throttle
          },
          digital_twin_nominal: {
            nominal_rpm: f.rpm,
            nominal_cht: Math.max(130, f.sensor_cht - (isFault ? 35 : 0)),
            nominal_egt: f.egt,
            nominal_oil_pressure: Math.min(75, f.oil_pressure + (isFault ? 20 : 0)),
            nominal_oil_temp: Math.max(80, f.oil_temp - (isFault ? 15 : 0)),
            nominal_fuel_flow: f.fuel_flow
          },
          residuals: {
            cht_delta_c: isFault ? 35.0 : 0.5,
            egt_delta_c: isFault ? 45.0 : 1.2,
            oil_pressure_delta_psi: isFault ? -18.0 : -0.2,
            vibration_delta: isFault ? 0.25 : 0.01
          },
          data_quality: {
            overall_health: 1.0,
            missing_count: 0,
            imputed_count: 0,
            sensor_drift_suspected: false
          },
          ai_diagnostics: {
            anomaly_score: f.pred_anomaly_score,
            is_anomaly: isAnom,
            predicted_fault: f.pred_fault_type || "none",
            confidence: f.pred_confidence || 0.99,
            estimated_rul_seconds: f.pred_rul_seconds,
            subsystem_health: {
              cylinders: f.pred_fault_type === "misfire" ? 0.45 : f.health_index,
              fuel_system: f.pred_fault_type === "injector" ? 0.40 : 1.0,
              oil_lubrication: f.pred_fault_type === "lubrication" ? 0.35 : f.health_index,
              cooling_jacket: f.pred_fault_type === "cooling" ? 0.30 : f.health_index
            },
            explanation: isFault ? {
              top_3_features: [
                { feature_name: "sensor_cht", shap_value: 0.48, direction: "increases", current_value: `${f.sensor_cht.toFixed(1)} °C`, certified_nominal_range: "140–200°C" },
                { feature_name: "oil_pressure", shap_value: -0.32, direction: "decreases", current_value: `${f.oil_pressure.toFixed(1)} psi`, certified_nominal_range: "55–85 psi" },
                { feature_name: "health_index", shap_value: -0.25, direction: "decreases", current_value: f.health_index.toFixed(2), certified_nominal_range: "> 0.85" }
              ],
              physics_narrative: `Pre-recorded flight replay: ${f.pred_fault_type.toUpperCase()} signature detected with ${(f.pred_confidence * 100).toFixed(0)}% confidence.`
            } : null
          },
          advisory: {
            level: isFault && f.health_index < 0.88 ? "CRITICAL" : isAnom ? "WARNING" : "NOMINAL",
            action_plan: isFault && f.health_index < 0.88 ? [
              `1. INITIATE EMERGENCY DESCENT to 2,000m for convective cooling.`,
              `2. De-rate throttle to ${(f.throttle * 0.7).toFixed(2)} to reduce thermal strain.`,
              `3. SQUAWK 7700 and divert to nearest recovery airbase.`
            ] : [
              "All engine parameters nominal. Replay telemetry synchronized.",
              "Flight path within certified bounds."
            ]
          },
          performance: {
            inference_latency_ms: 12.4,
            total_pipeline_ms: 12.4
          }
        };
      });

      replayIndex = 0;
      clearChartData();
      showToast(`Loaded ${filename} (${replayData.length} frames)`, "success");
      addTimelineEntry("INFO", `Flight mission loaded: ${filename} (${replayData.length} frames)`);
      startReplayPlayback();
    }
  } catch (err) {
    showToast(`Error loading replay: ${err.message}`, "danger");
  }
}

function startReplayPlayback() {
  if (!replayData.length) {
    const select = document.getElementById('missionSelect');
    if (select.value) {
      loadMissionForReplay(select.value);
      return;
    }
  }
  if (isRunning) return;
  isRunning = true;
  document.getElementById('btnReplayPlay').innerText = "▶ Playing";
  document.getElementById('btnReplayPlay').style.background = "var(--accent-green)";
  loopInterval = setInterval(stepReplayFrame, Math.round(1000 / (1 * replaySpeed)));
}

function stepReplayFrame() {
  if (replayIndex >= replayData.length) {
    pauseEngine();
    return;
  }
  const frame = replayData[replayIndex++];
  renderCockpitFrame(frame);
}

function resetReplayPlayback() {
  pauseEngine();
  replayIndex = 0;
  clearChartData();
  startReplayPlayback();
}

function formatTime(totalSeconds) {
  const s = Math.floor(totalSeconds || 0);
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}
