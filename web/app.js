"use strict";

const state = {
  status: null,
  benchmark: null,
  mode: "correct",
  selectedExample: "true_if_the",
  routing: [],
  activeVisit: 0,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number).toLocaleString() : "—";
}

function formatPercent(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
}

function formatProbability(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (number === 0) return "0";
  if (number < 0.0001) return number.toExponential(2);
  if (number < 0.01) return number.toFixed(5);
  return number.toFixed(3);
}

function formatBytes(value) {
  let number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let unit = 0;
  while (number >= 1024 && unit < units.length - 1) {
    number /= 1024;
    unit += 1;
  }
  return `${number.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

function modeShortLabel(mode) {
  return {
    correct: "Correct",
    shuffled: "Shuffled",
    semantic_opposite: "Semantic opposite",
    random_frozen: "Random frozen",
    zero: "Zero value",
    hash_only: "Hash only",
  }[mode] || mode;
}

function setLoading(loading) {
  document.body.classList.toggle("loading", loading);
  $("runButton").disabled = loading;
  $("compareButton").disabled = loading;
  $("runButton").querySelector("span").textContent = loading ? "Running…" : "Run trace";
}

let toastTimer = null;
function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("show"), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.message || payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function renderModes() {
  $("modeGrid").innerHTML = state.status.modes
    .map(
      (item) => `
        <button class="mode-button ${item.id === state.mode ? "active" : ""}"
                type="button" data-mode="${escapeHtml(item.id)}">
          ${escapeHtml(item.label)}
        </button>`,
    )
    .join("");
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      renderModes();
    });
  });
}

function renderExamples() {
  const select = $("exampleSelect");
  select.innerHTML = '<option value="">Custom prompt</option>' + state.status.examples
    .map(
      (item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title)} · “${escapeHtml(item.display_prompt)}” → ${escapeHtml(item.target)}</option>`,
    )
    .join("");
  select.value = state.selectedExample;
  applyExample(state.selectedExample);
}

function applyExample(exampleId) {
  const item = state.status.examples.find((record) => record.id === exampleId);
  if (!item) {
    state.selectedExample = "";
    return;
  }
  state.selectedExample = item.id;
  $("promptInput").value = item.display_prompt;
  $("targetInput").value = item.target;
}

function renderHeadline() {
  const headline = state.benchmark.headline;
  const cards = [
    [formatPercent(headline.active_conditional_fraction, 2), "conditional tissue active"],
    [`${headline.phrase_bank_compression_ratio.toFixed(2)}×`, "phrase-memory compression"],
    [formatPercent(headline.perplexity_reduction_vs_raw, 2), "perplexity reduction vs raw rows"],
    [`${headline.local_cards_vs_global_seed_wins}/${headline.local_cards_vs_global_seed_count}`, "MACSL wins vs global card"],
  ];
  $("headlineMetrics").innerHTML = cards
    .map(([value, label]) => `<div class="metric-chip"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`)
    .join("");
}

function renderBenchmark() {
  const headline = state.benchmark.headline;
  const benchmarkCards = [
    [`${headline.phrase_rows.toLocaleString()}`, "frozen phrase rows", "1,000 bigrams and 1,000 trigrams"],
    [`${headline.definition_rows.toLocaleString()}`, "definition vectors", "frozen official-Qwen layer-2 reference memory"],
    [formatPercent(headline.active_vocabulary_fraction, 1), "vocabulary active", "clustered sparse candidate scoring per token"],
    [formatPercent(headline.active_conditional_fraction, 2), "conditional capacity active", "top-2 experts and top-2 branches per expert"],
  ];
  $("benchmarkCards").innerHTML = benchmarkCards
    .map(
      ([value, label, note]) => `<div class="benchmark-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(note)}</p></div>`,
    )
    .join("");

  const methods = state.benchmark.full_architecture;
  $("benchmarkTable").querySelector("tbody").innerHTML = methods
    .map(
      (method) => `
        <tr class="${method.id === "centered_macsl_vivere32_r8" ? "selected-row" : ""}">
          <td>${escapeHtml(method.label)}</td>
          <td>${method.nll_mean.toFixed(5)} ± ${method.nll_std.toFixed(5)}</td>
          <td>${method.perplexity_mean.toFixed(1)}</td>
          <td>${formatPercent(method.sparse_accuracy_mean, 2)}</td>
          <td>${formatBytes(method.phrase_bank_bytes)}</td>
          <td>${method.latency_median_ms_mean.toFixed(2)} ms</td>
        </tr>`,
    )
    .join("");

  const maxBytes = Math.max(...methods.map((method) => method.phrase_bank_bytes));
  $("benchmarkBars").innerHTML = methods
    .map(
      (method) => `
        <div class="benchmark-bar">
          <span>${escapeHtml(method.label)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, (method.phrase_bank_bytes / maxBytes) * 100)}%"></div></div>
          <b>${formatBytes(method.phrase_bank_bytes)}</b>
        </div>`,
    )
    .join("");
}

function renderInitialResources() {
  const recipient = state.status.recipient;
  const memory = state.status.memory;
  const cards = [
    [formatInteger(recipient.stored_parameters), "stored recipient parameters"],
    [formatInteger(recipient.active_conditional_parameters_per_token), "active conditional / token"],
    [formatPercent(recipient.conditional_active_fraction, 2), "conditional activity"],
    [formatBytes(memory.stored_bytes), "VIVERE phrase-bank payload"],
    ["OFF", "Qwen donor at runtime"],
  ];
  $("resourceBand").innerHTML = cards
    .map(([value, label]) => `<div class="resource-card"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`)
    .join("");
}

function renderTokens(tokens, phraseHits) {
  const hitPositions = new Set((phraseHits || []).map((hit) => Number(hit.position)));
  $("tokenRibbon").innerHTML = tokens
    .map(
      (token, index) => `<span class="token-chip ${hitPositions.has(index) ? "hit" : ""}" title="position ${index}">${escapeHtml(token)}</span>`,
    )
    .join("");
}

function renderMemoryTrace(hits, definitions) {
  $("hitCount").textContent = `${hits.length} hit${hits.length === 1 ? "" : "s"}`;
  if (!hits.length) {
    $("memoryTrace").className = "memory-trace empty-state";
    $("memoryTrace").textContent = "No exact phrase hit. The recipient used deterministic hash fallback at eligible positions.";
  } else {
    $("memoryTrace").className = "memory-trace";
    $("memoryTrace").innerHTML = hits
      .map((hit) => {
        const value = hit.value_row === null || hit.value_row === undefined
          ? (hit.exact_enabled ? "synthetic / zero" : "hash fallback")
          : `row ${hit.value_row.toLocaleString()}`;
        return `
          <div class="memory-card">
            <header>
              <strong>“${escapeHtml(hit.phrase || `row ${hit.address_row}`)}”</strong>
              <small>${hit.order}-gram · position ${hit.position}</small>
            </header>
            <div class="memory-grid">
              <div class="memory-stat">Address<b>${hit.address_row.toLocaleString()}</b></div>
              <div class="memory-stat">Retrieved value<b>${escapeHtml(value)}</b></div>
              <div class="memory-stat">MACSL card<b>${hit.card_id ?? "—"}</b></div>
              <div class="memory-stat">Early / late norm<b>${hit.early_norm.toFixed(2)} / ${hit.late_norm.toFixed(2)}</b></div>
            </div>
          </div>`;
      })
      .join("");
  }

  $("definitionCount").textContent = String(definitions.length);
  $("definitionTrace").innerHTML = definitions.length
    ? definitions
        .map(
          (item) => `
            <div class="definition-item">
              <b>${escapeHtml(item.token)}</b> · row ${item.row}<br>
              ${escapeHtml(item.text || item.sense_id)}
            </div>`,
        )
        .join("")
    : '<div class="definition-item">No definition-memory rows were read for this prompt.</div>';
}

function renderCandidates(candidates) {
  if (!candidates.length) {
    $("candidateList").innerHTML = '<div class="empty-state">No sparse candidates returned.</div>';
    return;
  }
  const maximum = Math.max(...candidates.map((item) => item.candidate_probability), 1e-12);
  $("candidateList").innerHTML = candidates
    .map(
      (item) => `
        <div class="candidate-row">
          <span>#${item.rank}</span>
          <span class="candidate-token">${escapeHtml(item.token)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, (item.candidate_probability / maximum) * 100)}%"></div></div>
          <span class="candidate-prob">${formatPercent(item.candidate_probability, 1)}</span>
        </div>`,
    )
    .join("");
}

function renderTarget(target) {
  if (!target) {
    $("targetStrip").innerHTML = `
      <div class="target-metric"><strong>—</strong><span>No expected token supplied</span></div>
      <div class="target-metric"><strong>—</strong><span>Full-vocab probability</span></div>
      <div class="target-metric"><strong>—</strong><span>Negative log likelihood</span></div>
      <div class="target-metric"><strong>—</strong><span>Full-vocab rank</span></div>`;
    return;
  }
  $("targetStrip").innerHTML = `
    <div class="target-metric"><strong>${escapeHtml(target.token)}</strong><span>Expected token</span></div>
    <div class="target-metric"><strong>${formatProbability(target.probability)}</strong><span>Full-vocab probability</span></div>
    <div class="target-metric"><strong>${target.nll.toFixed(3)}</strong><span>Negative log likelihood</span></div>
    <div class="target-metric"><strong>#${target.rank.toLocaleString()}</strong><span>Full-vocab rank</span></div>`;
}

function renderRoutingVisit(index) {
  state.activeVisit = index;
  document.querySelectorAll(".visit-tab").forEach((tab, tabIndex) => {
    tab.classList.toggle("active", tabIndex === index);
  });
  const visit = state.routing[index];
  if (!visit) {
    $("routingCanvas").innerHTML = '<div class="empty-state">No routing trace returned.</div>';
    return;
  }
  const activeExperts = new Map(visit.selected_experts.map((item) => [item.expert, item]));
  const expertGrid = Array.from({ length: state.status.recipient.expert_count }, (_, expert) => `
    <div class="expert-node ${activeExperts.has(expert) ? "active" : ""}" title="Expert ${expert}">
      E${expert}
    </div>`).join("");
  const detail = visit.selected_experts
    .map((expert) => {
      const activeBranches = new Map(expert.branches.map((branch) => [branch.branch, branch]));
      const branches = Array.from({ length: state.status.recipient.branches_per_expert }, (_, branch) => {
        const record = activeBranches.get(branch);
        return `<div class="branch-node ${record ? "active" : ""}" title="Branch ${branch}${record ? ` · weight ${record.weight.toFixed(3)}` : ""}">B${branch}</div>`;
      }).join("");
      return `
        <div class="expert-detail">
          <strong>Expert ${expert.expert}</strong> <small>weight ${expert.weight.toFixed(3)}</small>
          <div class="branch-line">${branches}</div>
        </div>`;
    })
    .join("");
  const memory = visit.memory;
  $("routingCanvas").innerHTML = `
    <div class="expert-grid">${expertGrid}</div>
    <div class="selected-detail">${detail}</div>
    <div class="routing-meta">
      <div class="memory-stat">Early gate<b>${memory.early_gate.toFixed(3)}</b></div>
      <div class="memory-stat">Late gate<b>${memory.late_gate.toFixed(3)}</b></div>
      <div class="memory-stat">Exact / hash reads<b>${memory.exact_hits} / ${memory.hash_reads}</b></div>
      <div class="memory-stat">Relative change<b>${visit.relative_change.toFixed(4)}</b></div>
    </div>`;
}

function renderRouting(routing) {
  state.routing = routing || [];
  $("visitBadge").textContent = `${state.routing.length} visit${state.routing.length === 1 ? "" : "s"}`;
  if (!state.routing.length) {
    $("visitTabs").innerHTML = "";
    $("routingCanvas").innerHTML = '<div class="empty-state">No routing trace returned.</div>';
    return;
  }
  $("visitTabs").innerHTML = state.routing
    .map(
      (visit, index) => `<button type="button" class="visit-tab" data-visit="${index}">R${visit.round} · B${visit.block}</button>`,
    )
    .join("");
  document.querySelectorAll(".visit-tab").forEach((tab) => {
    tab.addEventListener("click", () => renderRoutingVisit(Number(tab.dataset.visit)));
  });
  renderRoutingVisit(state.routing.length - 1);
}

function renderAnalyze(result) {
  const first = result.steps[0];
  $("generatedText").textContent = result.generated_text || first.decoded_input;
  $("latencyBadge").textContent = `${first.latency_ms.toFixed(2)} ms`;
  $("vocabFraction").textContent = `${formatPercent(first.active_vocabulary_fraction, 1)} vocabulary active`;
  renderTarget(first.target);
  renderCandidates(first.sparse_candidates);
  renderTokens(first.tokens, first.phrase_hits);
  renderMemoryTrace(first.phrase_hits, first.definition_hits);
  renderRouting(first.routing);

  const resource = result.resource_summary;
  const cards = [
    [formatInteger(resource.stored_parameters), "stored recipient parameters"],
    [formatInteger(resource.active_conditional_parameters_per_token), "active conditional / token"],
    [formatPercent(resource.conditional_active_fraction, 2), "conditional activity"],
    [formatBytes(resource.phrase_bank_bytes), "VIVERE phrase-bank payload"],
    [resource.donor_loaded ? "ON" : "OFF", "Qwen donor at runtime"],
  ];
  $("resourceBand").innerHTML = cards
    .map(([value, label]) => `<div class="resource-card"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`)
    .join("");
}

function retrievedValue(item) {
  const hit = item.phrase_hits?.[item.phrase_hits.length - 1];
  if (!hit) return item.mode === "hash_only" ? "hash" : "no exact hit";
  if (item.mode === "hash_only") return `hash @ address ${hit.address_row}`;
  if (hit.value_row === null || hit.value_row === undefined) return `${item.mode} @ address ${hit.address_row}`;
  return `row ${hit.value_row} · card ${hit.card_id ?? "—"}`;
}

function renderComparison(result) {
  const targetAvailable = result.modes.some((item) => item.target);
  const values = result.modes.map((item) => targetAvailable ? (item.target?.probability || 0) : item.jensen_shannon_to_correct);
  const maximum = Math.max(...values, 1e-12);
  $("comparisonChart").className = "comparison-chart";
  $("comparisonChart").innerHTML = result.modes
    .map((item, index) => {
      const value = values[index];
      const label = targetAvailable ? formatProbability(value) : item.jensen_shannon_to_correct.toFixed(5);
      return `
        <div class="comparison-row ${item.mode === "correct" ? "correct" : ""}">
          <span class="comparison-name">${escapeHtml(modeShortLabel(item.mode))}</span>
          <div class="compare-track"><div class="compare-fill" style="width:${Math.max(2, (value / maximum) * 100)}%"></div></div>
          <b>${escapeHtml(label)}</b>
        </div>`;
    })
    .join("");

  $("comparisonTable").querySelector("tbody").innerHTML = result.modes
    .map((item) => `
      <tr class="${item.mode === "correct" ? "selected-row" : ""}">
        <td>${escapeHtml(item.label)}</td>
        <td>${item.target ? formatProbability(item.target.probability) : "—"}</td>
        <td>${item.target ? item.target.nll.toFixed(4) : "—"}</td>
        <td>${item.target ? `#${item.target.rank.toLocaleString()}` : "—"}</td>
        <td>${item.hidden_cosine_to_correct.toFixed(6)}</td>
        <td>${item.jensen_shannon_to_correct.toFixed(6)}</td>
        <td><code>${escapeHtml(retrievedValue(item))}</code></td>
      </tr>`)
    .join("");
  $("comparisonSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

function requestPayload() {
  const exampleId = $("exampleSelect").value;
  return {
    example_id: exampleId || undefined,
    prompt: exampleId ? undefined : $("promptInput").value,
    target: exampleId ? undefined : $("targetInput").value,
    mode: state.mode,
    max_new_tokens: Number($("generateSelect").value),
  };
}

async function runTrace() {
  setLoading(true);
  try {
    const result = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify(requestPayload()),
    });
    renderAnalyze(result);
  } catch (error) {
    toast(`Trace failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function compareAll() {
  setLoading(true);
  try {
    const payload = requestPayload();
    delete payload.mode;
    delete payload.max_new_tokens;
    const result = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderComparison(result);
  } catch (error) {
    toast(`Comparison failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function initialize() {
  try {
    const [status, benchmark] = await Promise.all([
      api("/api/status"),
      api("/api/benchmark"),
    ]);
    state.status = status;
    state.benchmark = benchmark;
    $("engineMode").textContent = `${status.engine} · ${status.environment.threads} CPU thread${status.environment.threads === 1 ? "" : "s"}`;
    $("claimBoundary").textContent = status.claim_boundary;
    renderModes();
    renderExamples();
    renderHeadline();
    renderBenchmark();
    renderInitialResources();

    $("exampleSelect").addEventListener("change", (event) => {
      const id = event.target.value;
      if (id) applyExample(id);
      else state.selectedExample = "";
    });
    $("promptInput").addEventListener("input", () => {
      if ($("exampleSelect").value) {
        $("exampleSelect").value = "";
        state.selectedExample = "";
      }
    });
    $("targetInput").addEventListener("input", () => {
      if ($("exampleSelect").value) {
        $("exampleSelect").value = "";
        state.selectedExample = "";
      }
    });
    $("runButton").addEventListener("click", runTrace);
    $("compareButton").addEventListener("click", compareAll);

    await runTrace();
  } catch (error) {
    $("engineMode").textContent = "Engine unavailable";
    toast(`Initialization failed: ${error.message}`);
  }
}

window.addEventListener("DOMContentLoaded", initialize);
