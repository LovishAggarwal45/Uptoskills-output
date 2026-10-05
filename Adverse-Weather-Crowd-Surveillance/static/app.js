// Adverse Weather Dynamic De-Hazing and Crowd Counting Dashboard
// Frontend Application Controller

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initImageUpload();
    initPresets();
    initVideoUpload();
    initVideoPresets();
    initLiveStream();
    initConfigForm();
    loadRecentLogs();
    
    // Auto-refresh logs every 10 seconds
    setInterval(loadRecentLogs, 10000);
});

// =========================================================================
// 1. TAB CONTROLLER
// =========================================================================
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add("active");
        });
    });
}

// =========================================================================
// 2. TELEMETRY & UI UPDATERS
// =========================================================================
function updateTelemetry(t) {
    // 1. Crowd Count
    const countEl = document.getElementById("metric-count");
    if (countEl) countEl.innerText = Math.round(t.crowd_count).toLocaleString();
    
    const boxCount = t.bounding_box_count !== undefined ? t.bounding_box_count : (t.bounding_boxes ? t.bounding_boxes.length : 0);
    const countSubEl = document.getElementById("metric-count-sub");
    if (countSubEl) countSubEl.innerText = `Peak: ${t.density_peak.toFixed(4)} | Boxes: ${boxCount} | Cap: ${t.max_crowd_capacity}`;

    const restSub = document.getElementById("panel-rest-sub");
    if (restSub) restSub.innerText = `Emerald Bounding Boxes (${boxCount} localized)`;
    
    // 2. Condition & Visibility
    const condBadge = document.getElementById("badge-condition");
    if (condBadge) {
        condBadge.className = `badge-condition badge-${t.detected_condition.toLowerCase()}`;
        condBadge.innerText = t.detected_condition.toUpperCase();
    }
    
    const visEl = document.getElementById("metric-vis");
    if (visEl) visEl.innerText = t.visibility_index.toFixed(3);
    
    const visSubEl = document.getElementById("metric-vis-sub");
    if (visSubEl) {
        const [r, g, b] = t.airlight_A;
        visSubEl.innerText = `Airlight A: [${r.toFixed(2)}, ${g.toFixed(2)}, ${b.toFixed(2)}]`;
    }
    
    // 3. Degradation Explanation
    const degEl = document.getElementById("metric-degrade-exp");
    if (degEl) degEl.innerText = t.degradation_explanation;
    
    // 4. Risk Level & Multiplier
    const riskEl = document.getElementById("metric-risk");
    if (riskEl) riskEl.innerText = `${(t.risk_score * 100).toFixed(0)}%`;
    
    const riskSubEl = document.getElementById("metric-risk-sub");
    if (riskSubEl) riskSubEl.innerText = `Level: ${t.risk_level} (w_mult: ${t.weather_multiplier}x)`;
    
    const riskBar = document.getElementById("risk-bar");
    if (riskBar) {
        const pct = Math.min(100, Math.round(t.risk_score * 100));
        riskBar.style.width = `${pct}%`;
        if (t.risk_level === "CRITICAL") riskBar.style.backgroundColor = "#ef4444";
        else if (t.risk_level === "HIGH") riskBar.style.backgroundColor = "#f97316";
        else if (t.risk_level === "MEDIUM") riskBar.style.backgroundColor = "#f59e0b";
        else riskBar.style.backgroundColor = "#10b981";
    }
    
    // 5. Latency & Location
    const latEl = document.getElementById("metric-latency");
    if (latEl) latEl.innerText = `${t.latency_ms} ms`;
    
    const latSubEl = document.getElementById("metric-latency-sub");
    if (latSubEl) latSubEl.innerText = `${t.fps} FPS | Loc: ${t.location}`;
    
    // 6. Header Alert Banner
    const banner = document.getElementById("alert-banner");
    const bannerTxt = document.getElementById("alert-banner-text");
    if (banner && bannerTxt) {
        banner.className = `alert-banner ${t.risk_level}`;
        bannerTxt.innerHTML = `<strong>[${t.risk_level}] ${t.alert_status}</strong> &mdash; ${t.location} | Count: ${Math.round(t.crowd_count)} / ${t.max_crowd_capacity} (Risk: ${t.risk_score.toFixed(2)})`;
    }
}

function updateVisualPanels(visuals) {
    if (visuals.original_frame) {
        document.getElementById("img-original").src = visuals.original_frame;
        document.getElementById("panel-orig-sub").innerText = "Feed Resolution: Restored Native";
    }
    if (visuals.transmission_map) {
        document.getElementById("img-transmission").src = visuals.transmission_map;
        document.getElementById("panel-trans-sub").innerText = "Optical Prior: PhysicsGuidedDehazeNet";
    }
    if (visuals.restored_frame) {
        document.getElementById("img-restored").src = visuals.restored_frame;
        document.getElementById("panel-rest-sub").innerText = "Dehazed Physics-Refined Output";
    }
    if (visuals.density_heatmap) {
        document.getElementById("img-density").src = visuals.density_heatmap;
        document.getElementById("panel-dens-sub").innerText = "CSRNet Dilated Kernel Regression";
    }
}

// =========================================================================
// 3. SINGLE FRAME IMAGE PROCESSING
// =========================================================================
function initImageUpload() {
    const form = document.getElementById("image-upload-form");
    const fileInput = document.getElementById("image-file-input");
    const submitBtn = document.getElementById("btn-submit-image");
    const statusText = document.getElementById("image-status-text");

    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!fileInput.files || fileInput.files.length === 0) {
            alert("Please select an image file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        const loc = document.getElementById("image-location-override").value;
        if (loc) formData.append("location", loc);

        submitBtn.disabled = true;
        statusText.innerText = "Analyzing degradation, inverting scattering, & regressing crowd density...";

        try {
            const resp = await fetch("/api/process/image", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Server failed to process image.");
            }

            const data = await resp.json();
            updateTelemetry(data.telemetry);
            updateVisualPanels(data.visuals);
            statusText.innerText = `Processed in ${data.telemetry.latency_ms} ms. Condition: ${data.telemetry.detected_condition.toUpperCase()}`;
            loadRecentLogs();
        } catch (err) {
            console.error(err);
            statusText.innerText = `Error: ${err.message}`;
        } finally {
            submitBtn.disabled = false;
        }
    });
}

// =========================================================================
// 4. PRESET SAMPLES (Fog, Rain, Glare, Clear)
// =========================================================================
function initPresets() {
    const presetBtns = document.querySelectorAll(".btn-preset");
    const statusText = document.getElementById("image-status-text");

    presetBtns.forEach(btn => {
        btn.addEventListener("click", async () => {
            const presetName = btn.getAttribute("data-preset");
            statusText.innerText = `Loading verified preset: ${presetName}...`;

            try {
                const resp = await fetch(`/api/process/preset?name=${presetName}`, {
                    method: "POST"
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || "Preset processing failed.");
                }

                const data = await resp.json();
                updateTelemetry(data.telemetry);
                updateVisualPanels(data.visuals);
                statusText.innerText = `Loaded preset: ${presetName} (${data.telemetry.latency_ms} ms)`;
                loadRecentLogs();
            } catch (err) {
                console.error(err);
                statusText.innerText = `Error loading preset: ${err.message}`;
            }
        });
    });
}

// =========================================================================
// 5. VIDEO BATCH PROCESSING
// =========================================================================
function initVideoUpload() {
    const form = document.getElementById("video-upload-form");
    const fileInput = document.getElementById("video-file-input");
    const submitBtn = document.getElementById("btn-submit-video");
    const statusText = document.getElementById("video-status-text");
    const playerContainer = document.getElementById("video-result-container");
    const videoPlayer = document.getElementById("video-player");

    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!fileInput.files || fileInput.files.length === 0) {
            alert("Please select a video file (MP4/AVI).");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        const interval = document.getElementById("video-sampling-interval").value;
        formData.append("sampling_interval", interval);
        const loc = document.getElementById("video-location-override").value;
        if (loc) formData.append("location", loc);

        submitBtn.disabled = true;
        statusText.innerText = `Processing video periodically (every ${interval} frames)... Please wait.`;
        if (playerContainer) playerContainer.style.display = "none";

        try {
            const resp = await fetch("/api/process/video", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Video processing failed.");
            }

            const data = await resp.json();
            statusText.innerText = `Completed ${data.processed_frames} sampled frames. Rendering video output...`;
            
            if (videoPlayer && playerContainer) {
                videoPlayer.src = data.video_url;
                playerContainer.style.display = "block";
                videoPlayer.load();
                videoPlayer.play();
            }
            loadRecentLogs();
        } catch (err) {
            console.error(err);
            statusText.innerText = `Error: ${err.message}`;
        } finally {
            submitBtn.disabled = false;
        }
    });
}

function initVideoPresets() {
    const presetBtns = document.querySelectorAll(".btn-video-preset");
    const statusText = document.getElementById("video-status-text");
    const playerContainer = document.getElementById("video-result-container");
    const videoPlayer = document.getElementById("video-player");

    presetBtns.forEach(btn => {
        btn.addEventListener("click", async () => {
            const presetName = btn.getAttribute("data-video-preset");
            const interval = document.getElementById("video-sampling-interval").value || 3;
            statusText.innerText = `Processing preset video: ${presetName}... Please wait.`;
            if (playerContainer) playerContainer.style.display = "none";

            try {
                const resp = await fetch(`/api/process/video-preset?name=${presetName}&sampling_interval=${interval}`, {
                    method: "POST"
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || "Video preset processing failed.");
                }

                const data = await resp.json();
                statusText.innerText = `Processed ${data.processed_frames} sampled frames from preset ${presetName}!`;

                if (videoPlayer && playerContainer) {
                    videoPlayer.src = data.video_url;
                    playerContainer.style.display = "block";
                    videoPlayer.load();
                    videoPlayer.play();
                }
                loadRecentLogs();
            } catch (err) {
                console.error(err);
                statusText.innerText = `Error: ${err.message}`;
            }
        });
    });
}

// =========================================================================
// 6. LIVE CAMERA STREAM CONTROLLER
// =========================================================================
function initLiveStream() {
    const liveImg = document.getElementById("live-stream-img");
    const toggleBtn = document.getElementById("btn-toggle-stream");
    let isStreaming = false;

    if (!toggleBtn || !liveImg) return;

    toggleBtn.addEventListener("click", () => {
        if (!isStreaming) {
            liveImg.src = "/api/stream/feed";
            liveImg.style.display = "block";
            toggleBtn.innerText = "Pause Live Feed";
            toggleBtn.classList.remove("btn-primary");
            toggleBtn.classList.add("btn-secondary");
            isStreaming = true;
        } else {
            liveImg.src = "";
            liveImg.style.display = "none";
            toggleBtn.innerText = "Start Live Surveillance Stream";
            toggleBtn.classList.remove("btn-secondary");
            toggleBtn.classList.add("btn-primary");
            isStreaming = false;
        }
    });
}

// =========================================================================
// 7. SURVEILLANCE CONFIGURATION CONTROLLER
// =========================================================================
function initConfigForm() {
    const form = document.getElementById("config-form");
    const statusText = document.getElementById("config-status-text");

    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            location: document.getElementById("cfg-location").value,
            max_crowd_capacity: parseFloat(document.getElementById("cfg-capacity").value),
            sampling_interval: parseInt(document.getElementById("cfg-interval").value),
            weather_multipliers: {
                clear: parseFloat(document.getElementById("cfg-mult-clear").value),
                fog: parseFloat(document.getElementById("cfg-mult-fog").value),
                rain: parseFloat(document.getElementById("cfg-mult-rain").value),
                glare: parseFloat(document.getElementById("cfg-mult-glare").value)
            }
        };

        try {
            const resp = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) throw new Error("Failed to update configuration.");
            statusText.innerText = "Configuration updated successfully!";
            setTimeout(() => { statusText.innerText = ""; }, 3000);
        } catch (err) {
            console.error(err);
            statusText.innerText = `Error: ${err.message}`;
        }
    });
}

// =========================================================================
// 8. AUDIT LOG VIEWER & CSV SYNCHRONIZATION
// =========================================================================
async function loadRecentLogs() {
    const tbody = document.getElementById("log-table-body");
    if (!tbody) return;

    try {
        const resp = await fetch("/api/logs?limit=25");
        if (!resp.ok) return;

        const data = await resp.json();
        const logs = data.logs;

        if (!logs || logs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No surveillance events recorded yet. Upload a frame or run a sample.</td></tr>`;
            return;
        }

        tbody.innerHTML = "";
        logs.reverse().forEach(row => {
            const tr = document.createElement("tr");

            // Format timestamp nicely
            let timeStr = row.timestamp;
            try {
                const d = new Date(row.timestamp);
                timeStr = d.toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
            } catch (e) {}

            // Alert status styling
            let statusClass = "NORMAL";
            if (row.alert_status.includes("CAPACITY")) statusClass = "CAPACITY";
            else if (row.alert_status.includes("ELEVATED")) statusClass = "ELEVATED";
            else if (row.alert_status.includes("CAUTION")) statusClass = "CAUTION";

            tr.innerHTML = `
                <td>${timeStr}</td>
                <td><strong>${escapeHtml(row.location)}</strong></td>
                <td><span class="badge-condition badge-${(row.detected_condition || 'clear').toLowerCase()}">${(row.detected_condition || '').toUpperCase()}</span></td>
                <td style="font-weight: 700; color: #fff;">${row.crowd_count}</td>
                <td>${row.visibility_index}</td>
                <td>${row.risk_score}</td>
                <td><span class="status-tag ${statusClass}">${escapeHtml(row.alert_status)}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Error fetching logs:", err);
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
