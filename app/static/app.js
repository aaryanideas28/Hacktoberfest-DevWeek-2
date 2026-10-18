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
        const estTokens = Math.floor(text.length / 4);
        document.getElementById('input-token-count').innerText = `${estTokens} tokens`;
    });
}

// Run compression via API
async function runCompression() {
    const prompt = document.getElementById('compress-input').value;
    const mode = document.getElementById('compress-mode').value;
    const btn = document.getElementById('btn-run-compress');

    if (!prompt.trim()) {
        alert("Please enter a prompt to compress.");
        return;
    }

    btn.disabled = true;
    btn.innerText = "Compressing...";

    try {
        const res = await fetch(`/api/v1/compress?mode=${mode}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });

        const data = await res.json();
        if (res.ok) {
            document.getElementById('compress-output').value = data.compressed_prompt;
            document.getElementById('stat-orig').innerText = data.original_tokens;
            document.getElementById('stat-comp').innerText = data.compressed_tokens;
            const pct = data.original_tokens > 0 
                ? Math.round((data.tokens_saved / data.original_tokens) * 100) 
                : 0;
            document.getElementById('stat-saved').innerText = `${data.tokens_saved} (${pct}%)`;
        } else {
            alert(data.detail || "Error compressing prompt.");
        }
    } catch (err) {
        console.error(err);
        alert("Failed to connect to backend server.");
    } finally {
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
    btn.innerText = "Analyzing Context...";

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
    if (text) navigator.clipboard.writeText(text);
}

function copyHandoff() {
    const text = document.getElementById('handoff-output').value;
    if (text) navigator.clipboard.writeText(text);
}
