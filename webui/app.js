"use strict";

// Small DOM helper: builds elements with textContent only (no innerHTML), so file contents cannot inject markup.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function predictBlob(blob) {
  return api("/api/predict", { method: "POST", headers: { "Content-Type": blob.type || "application/octet-stream" }, body: blob });
}

async function refreshBadge() {
  const badge = document.getElementById("modelBadge");
  try {
    const data = await api("/api/models");
    badge.textContent = data.active ? `Model: ${data.active}` : "No trained model found";
  } catch (error) {
    badge.textContent = "Server unavailable";
  }
}

function showResult(container, result) {
  container.className = `result ${result.label === 1 ? "inside" : "outside"}`;
  container.replaceChildren(el("div", { class: "verdict" }, result.text));
  const parts = [];
  if (result.score !== null && result.score !== undefined) parts.push(`${result.score_kind}: ${result.score.toFixed(2)}`);
  parts.push(`model ${result.model}`, `${result.elapsed_ms} ms`);
  container.append(el("div", { class: "detail" }, parts.join(" | ")));
}

function showError(container, message) {
  container.className = "result error";
  container.replaceChildren(el("div", {}, message));
}

/* ---------- Predict page ---------- */
function initPredictPage() {
  const panes = { upload: document.getElementById("paneUpload"), camera: document.getElementById("paneCamera") };
  const tabs = { upload: document.getElementById("tabUpload"), camera: document.getElementById("tabCamera") };
  function showTab(name) {
    for (const key of Object.keys(panes)) {
      panes[key].classList.toggle("hidden", key !== name);
      tabs[key].classList.toggle("active", key === name);
    }
    if (name !== "camera") stopCamera();
  }
  tabs.upload.addEventListener("click", () => showTab("upload"));
  tabs.camera.addEventListener("click", () => showTab("camera"));

  // Upload (batch): one request per photo, so a bad file never blocks the others
  const input = document.getElementById("fileInput");
  const dropzone = document.getElementById("dropzone");
  const batchTable = document.getElementById("batchTable");
  const batchBody = document.getElementById("batchBody");
  const batchControls = document.getElementById("batchControls");
  const batchSummary = document.getElementById("batchSummary");
  const exportButton = document.getElementById("exportButton");
  const clearButton = document.getElementById("clearButton");
  const items = [];
  let working = false;
  let nextId = 1;

  function paint(item) {
    const [, nameCell, verdictCell, detailCell] = item.row.children;
    nameCell.replaceChildren(el("div", {}, item.file.name), el("small", { class: "muted" }, `${(item.file.size / 1024).toFixed(0)} KB`));
    detailCell.replaceChildren();
    if (item.status === "done") {
      const r = item.result;
      verdictCell.replaceChildren(el("span", { class: `badge ${r.label === 1 ? "inside" : "outside"}` }, r.text));
      const parts = [];
      if (r.score !== null && r.score !== undefined) parts.push(`${r.score_kind}: ${r.score.toFixed(2)}`);
      parts.push(`${r.elapsed_ms} ms`);
      detailCell.textContent = parts.join(" | ");
    } else if (item.status === "error") {
      verdictCell.replaceChildren(el("span", { class: "badge error" }, "Failed"));
      detailCell.textContent = item.error;
    } else {
      verdictCell.textContent = item.status === "running" ? "Classifying..." : "Waiting";
    }
  }

  function buildRow(item) {
    const link = el("a", { href: item.url, target: "_blank", rel: "noopener" },
      el("img", { class: "thumb", src: item.url, alt: item.file.name, loading: "lazy" }));
    item.row = el("tr", {}, el("td", {}, link), el("td"), el("td"), el("td"));
    paint(item);
    return item.row;
  }

  function refreshSummary() {
    const done = items.filter((item) => item.status === "done");
    const inside = done.filter((item) => item.result.label === 1).length;
    const failed = items.filter((item) => item.status === "error").length;
    const pending = items.length - done.length - failed;
    batchTable.classList.toggle("hidden", items.length === 0);
    batchControls.classList.toggle("hidden", items.length === 0);
    exportButton.disabled = done.length === 0;
    const parts = [`${items.length} photo${items.length === 1 ? "" : "s"}`, `${inside} inside`, `${done.length - inside} outside`];
    if (failed) parts.push(`${failed} failed`);
    if (pending) parts.push(`${pending} still pending`);
    batchSummary.textContent = parts.join(" | ");
  }

  async function processQueue() {
    if (working) return;
    working = true;
    try {
      let item;
      while ((item = items.find((entry) => entry.status === "queued"))) {
        item.status = "running";
        paint(item);
        try {
          item.result = await predictBlob(item.file);  // raw bytes only: the file name is never sent
          item.status = "done";
        } catch (error) {
          item.error = error.message;
          item.status = "error";
        }
        paint(item);
        refreshSummary();
      }
    } finally {
      working = false;
    }
  }

  function addFiles(fileList) {
    for (const file of Array.from(fileList)) {
      const item = { id: nextId++, file, url: URL.createObjectURL(file), status: "queued", result: null, error: null };
      items.push(item);
      batchBody.append(buildRow(item));
    }
    refreshSummary();
    processQueue();
  }

  function csvCell(value) {
    let text = value === null || value === undefined ? "" : String(value);
    if (/^[=+\-@]/.test(text)) text = `'${text}`;  // stop spreadsheets from treating file names as formulas
    return `"${text.replace(/"/g, '""')}"`;
  }

  function exportCsv() {
    const header = ["file_name", "prediction", "label", "score", "score_kind", "model", "status", "error"];
    const rows = items.map((item) => {
      const r = item.result || {};
      return [item.file.name, r.text, r.label, r.score, r.score_kind, r.model, item.status, item.error].map(csvCell).join(",");
    });
    const blob = new Blob(["\ufeff" + [header.join(","), ...rows].join("\r\n")], { type: "text/csv;charset=utf-8" });
    const link = el("a", { href: URL.createObjectURL(blob), download: "predictions.csv" });
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function clearAll() {
    for (const item of items) URL.revokeObjectURL(item.url);
    items.length = 0;
    batchBody.replaceChildren();
    refreshSummary();
  }

  input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
  dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("over"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("over");
    addFiles(event.dataTransfer.files);
  });
  exportButton.addEventListener("click", exportCsv);
  clearButton.addEventListener("click", clearAll);

  // Camera
  const video = document.getElementById("video");
  const canvas = document.getElementById("grab");
  const startButton = document.getElementById("camStart");
  const stopButton = document.getElementById("camStop");
  const intervalSelect = document.getElementById("camInterval");
  const cameraResult = document.getElementById("cameraResult");
  const stableLine = document.getElementById("cameraStable");
  const historyBar = document.getElementById("history");
  const WINDOW = 5;
  const MAX_WIDTH = 960;
  let stream = null, timer = null, running = false, recent = [];

  function grabFrame() {
    const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));  // lossless: no extra JPEG artefacts
  }

  function updateHistory(label) {
    recent.push(label);
    if (recent.length > WINDOW) recent.shift();
    historyBar.replaceChildren(...recent.map((value) => el("span", { class: value === 1 ? "inside" : "outside" })));
    const inside = recent.filter((value) => value === 1).length;
    const stableLabel = inside * 2 > recent.length ? "Inside" : "Outside";
    const votes = stableLabel === "Inside" ? inside : recent.length - inside;
    stableLine.textContent = `Stable: ${stableLabel} (${votes}/${recent.length} recent frames)`;
  }

  async function loop() {
    if (!running) return;
    const started = performance.now();
    try {
      if (video.videoWidth > 0) {
        const result = await predictBlob(await grabFrame());
        if (running) { showResult(cameraResult, result); updateHistory(result.label); }
      }
    } catch (error) {
      if (running) showError(cameraResult, error.message);
    }
    const wait = Math.max(0, Number(intervalSelect.value) - (performance.now() - started));
    timer = setTimeout(loop, wait);  // next frame starts only after the previous one finished
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError(cameraResult, "This browser cannot access a camera on this page. Open http://127.0.0.1:8000 or http://localhost:8000.");
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    } catch (error) {
      showError(cameraResult, `Camera unavailable: ${error.message}`);
      return;
    }
    video.srcObject = stream;
    await video.play();
    running = true;
    recent = [];
    startButton.disabled = true;
    stopButton.disabled = false;
    cameraResult.className = "result";
    cameraResult.textContent = "Starting...";
    loop();
  }

  function stopCamera() {
    running = false;
    clearTimeout(timer);
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    startButton.disabled = false;
    stopButton.disabled = true;
  }

  startButton.addEventListener("click", startCamera);
  stopButton.addEventListener("click", stopCamera);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopCamera(); });
  window.addEventListener("beforeunload", stopCamera);
}

/* ---------- Config page ---------- */
function table(headers, rows) {
  return el("table", {},
    el("thead", {}, el("tr", {}, ...headers.map((name) => el("th", {}, name)))),
    el("tbody", {}, ...rows.map((row) => el("tr", {}, ...row.map((cell) => el("td", {}, cell))))));
}

async function renderModels() {
  const list = document.getElementById("modelList");
  const data = await api("/api/models");
  if (!data.models.length) {
    list.replaceChildren(el("div", { class: "warn" }, "No pipeline.joblib found. Run training first, then reload this page."));
    return;
  }
  list.replaceChildren(...data.models.map((model) => {
    const title = el("strong", {}, model.id);
    if (model.is_best) title.append(el("span", { class: "tag" }, "best"));
    const lines = [];
    if (model.variant_id) lines.push(`variant: ${model.variant_id}`);
    if (model.trained_on_splits) lines.push(`trained on: ${[].concat(model.trained_on_splits).join(", ")}`);
    if (model.selected_using_splits) lines.push(`selected using: ${[].concat(model.selected_using_splits).join(", ")}`);
    lines.push(`${model.size_kb} KB`);
    const radio = el("input", { type: "radio", name: "model" });
    radio.checked = model.id === data.active;
    const row = el("label", { class: `model${model.id === data.active ? " active" : ""}` }, radio,
      el("div", {}, title, el("small", {}, lines.join(" | "))));
    radio.addEventListener("change", async () => {
      const message = document.getElementById("modelMessage");
      message.textContent = "Loading model...";
      try {
        await api("/api/models/select", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: model.id }) });
        message.textContent = `Now using ${model.id}.`;
      } catch (error) {
        message.textContent = error.message;
      }
      await Promise.all([renderModels(), renderPipeline(), renderDataset(), refreshBadge()]);
    });
    return row;
  }));
}

async function renderPipeline() {
  const box = document.getElementById("pipelineInfo");
  try {
    const data = await api("/api/models/active");
    if (!data.active) { box.textContent = "No active model."; return; }
    const steps = data.pipeline.steps.map((step) => [step.name, step.type, JSON.stringify(step.params)]);
    box.replaceChildren(
      table(["Step", "Type", "Parameters"], steps),
      el("details", {}, el("summary", {}, "Saved metadata (pipeline_metadata.json)"),
        el("pre", {}, JSON.stringify(data.metadata, null, 2))));
  } catch (error) {
    box.replaceChildren(el("div", { class: "warn" }, error.message));
  }
}

async function renderDataset() {
  const box = document.getElementById("datasetInfo");
  const data = await api("/api/dataset");
  if (!data.available) { box.replaceChildren(el("div", { class: "warn" }, data.message)); return; }
  const nodes = [];
  if (data.matches_active_model === false) {
    nodes.push(el("div", { class: "warn" }, "images.csv has changed since the active model was trained. Retrain before trusting the counts below."));
  } else if (data.matches_active_model === true) {
    nodes.push(el("p", { class: "muted" }, "images.csv matches the one recorded when the active model was trained."));
  }
  nodes.push(el("p", {}, `${data.total_images} images in ${data.total_groups} location groups (${data.dataset_dir}).`));
  nodes.push(table(["Split", "Inside images (groups)", "Outside images (groups)", "Total images"],
    data.summary.map((row) => [row.split, `${row.inside.images} (${row.inside.groups})`, `${row.outside.images} (${row.outside.groups})`, row.total_images])));
  nodes.push(el("h3", {}, "Location groups"));
  nodes.push(table(["Group", "Class", "Split", "Images", "Description", "Verified"],
    data.groups.map((g) => [g.group_id, g.class, g.split, g.images, g.description, g.boundary_verified])));
  box.replaceChildren(...nodes);
}

document.addEventListener("DOMContentLoaded", () => {
  refreshBadge();
  const page = document.body.dataset.page;
  if (page === "predict") initPredictPage();
  if (page === "config") {
    renderModels().catch((error) => { document.getElementById("modelList").textContent = error.message; });
    renderPipeline();
    renderDataset().catch((error) => { document.getElementById("datasetInfo").textContent = error.message; });
  }
});
