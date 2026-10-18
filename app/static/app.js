// Tab switcher
function switchTab(tabName) {
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    document.getElementById(`tab-${tabName}`).classList.add('active');
    document.getElementById(`tab-btn-${tabName}`).classList.add('active');
}

// Live token estimator for input box (~4 chars per token)
const compressInput = document.getElementById('compress-input');
if (compressInput) {
    compressInput.addEventListener('input', (e) => {
        const text = e.target.value.trim();
        const estTokens = text.length > 0 ? Math.max(1, Math.floor(text.length / 4)) : 0;
        document.getElementById('input-token-count').innerText = `${estTokens} tokens`;
    });
}

// Check system health on load to update UI badges
async function checkHealth() {
    try {
        const res = await fetch('/api/v1/health');
        if (res.ok) {
            const data = await res.json();
            const gemmaPill = document.getElementById('gemma-status');
            const sentryPill = document.getElementById('sentry-status');

            if (data.gemma_active) {
                gemmaPill.innerText = "🧠 Gemma 2: Connected";
                gemmaPill.classList.add("active-pill");
            } else {
                gemmaPill.innerText = "⚡ Gemma 2 2B: Local Entropy Engine";
            }

            if (data.sentry_active) {
                sentryPill.innerText = "📈 Sentry: Tracing Live";
            } else {
                sentryPill.innerText = "📊 Tracing: Ready";
            }
        }
    } catch (err) {
        console.warn("Health check unreachable, running standalone:", err);
    }
}
checkHealth();

// ---- helpers ----
async function postCompress(prompt, mode, dropRate, timeoutMs) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        const res = await fetch('/api/v1/compress', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, mode: mode, drop_rate: dropRate }),
            signal: ctrl.signal
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Error compressing prompt.");
        return data;
    } finally {
        clearTimeout(timer);
    }
}

function showResult(data, tag) {
    document.getElementById('compress-output').value = data.compressed_prompt;
    document.getElementById('stat-orig').innerText = data.original_tokens;
    document.getElementById('stat-comp').innerText = data.compressed_tokens;
    document.getElementById('stat-saved').innerText = `${data.tokens_saved} (${data.compression_ratio}%)`;
    document.getElementById('stat-engine').innerText = `Engine: ${data.engine_used}${tag ? ' · ' + tag : ''}`;
    document.getElementById('stat-latency').innerText = `Latency: ${data.latency_ms || '< 1'} ms`;

    let box = document.getElementById('compress-extras');
    if (!box) {
        box = document.createElement('div');
        box.id = 'compress-extras';
        box.style.cssText = 'margin-top:8px;font-size:12px;opacity:.85;line-height:1.4;white-space:pre-line';
        document.getElementById('compress-output').insertAdjacentElement('afterend', box);
    }
    const parts = [];
    if (data.note) parts.push('ℹ️ ' + data.note);
    if (data.removed_words && data.removed_words.length) parts.push('Removed: ' + data.removed_words.join(', '));
    box.textContent = parts.join('\n');
}

// Run compression: instant rule result first, then AI upgrade
async function runCompression() {
    const prompt = document.getElementById('compress-input').value;
    const mode = document.getElementById('compress-mode').value;
    const dropRate = parseFloat(document.getElementById('drop-rate').value || "0.40");
    const btn = document.getElementById('btn-run-compress');

    if (!prompt.trim()) {
        alert("Please enter a prompt to compress.");
        return;
    }

    btn.disabled = true;
    const t0 = Date.now();
    const tick = setInterval(() => {
        btn.innerText = `Compressing… ${Math.floor((Date.now() - t0) / 1000)}s`;
    }, 250);

    try {
        // Phase 1: instant, no API
        const quick = await postCompress(prompt, 'rule', dropRate, 15000);
        showResult(quick, mode === 'rule' ? '' : 'instant');

        // Phase 2: AI rewrite (only if an AI mode is selected)
        if (mode !== 'rule') {
            try {
                const ai = await postCompress(prompt, mode, dropRate, 25000);
                if (ai.compressed_tokens < quick.compressed_tokens) {
                    showResult(ai, 'AI upgrade');
                } else {
                    document.getElementById('stat-engine').innerText =
                        `Engine: ${quick.engine_used} · no safe AI gain`;
                }
            } catch (e) {
                document.getElementById('stat-engine').innerText =
                    `Engine: ${quick.engine_used} · AI step skipped (slow or unavailable)`;
            }
        }
    } catch (err) {
        console.error(err);
        alert(err.name === 'AbortError' ? "Request timed out." : (err.message || "Failed to connect to backend server."));
    } finally {
        clearInterval(tick);
        btn.disabled = false;
        btn.innerText = "Compress Prompt";
    }
}

// Run context handoff extraction via API
async function runHandoff() {
    const rawTranscript = document.getElementById('handoff-input').value;
    const projectName = document.getElementById('project-name-input').value || "Default-Project";
    const btn = document.getElementById('btn-run-handoff');

    if (!rawTranscript.trim()) {
        alert("Please enter conversation or log text.");
        return;
    }

    btn.disabled = true;
    btn.innerText = "Extracting Context...";

    try {
        const res = await fetch('/api/v1/handoff', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_name: projectName, raw_transcript: rawTranscript })
        });

        const data = await res.json();
        if (res.ok) {
            document.getElementById('handoff-output').value = data.formatted_markdown;
        } else {
            alert(data.detail || "Error extracting context.");
        }
    } catch (err) {
        console.error(err);
        alert("Failed to connect to backend server.");
    } finally {
        btn.disabled = false;
        btn.innerText = "Generate Handoff Artifact";
    }
}

// Clipboard copy helpers
function copyCompressed() {
    const text = document.getElementById('compress-output').value;
    if (text) {
        navigator.clipboard.writeText(text);
        const btn = document.getElementById('btn-copy-compress');
        btn.innerText = "✅ Copied!";
        setTimeout(() => { btn.innerText = "📋 Copy"; }, 1500);
    }
}

function copyHandoff() {
    const text = document.getElementById('handoff-output').value;
    if (text) {
        navigator.clipboard.writeText(text);
        const btn = document.getElementById('btn-copy-handoff');
        btn.innerText = "✅ Copied!";
        setTimeout(() => { btn.innerText = "📋 Copy for Next LLM"; }, 1500);
    }
}
