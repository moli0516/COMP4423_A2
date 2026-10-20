"use strict";
// Evaluate page. Relies on el() and api() from app.js (loaded first).

const SVG_NS = "http://www.w3.org/2000/svg";
const INSIDE_COLOUR = "#c0392b";
const OUTSIDE_COLOUR = "#2c6fbb";

function pct(value) { return `${(100 * value).toFixed(1)}%`; }

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function statCard(label, value, note) {
  return el("div", { class: "ev-stat" }, el("div", { class: "ev-value" }, value), el("div", { class: "ev-label" }, label), note ? el("small", { class: "muted" }, note) : "");
}

function confusionMatrix(metrics) {
  const m = metrics.confusion_matrix;
  const names = ["outside", "inside"];
  const rows = [0, 1].map((truth) => {
    const total = m[truth][0] + m[truth][1];
    const cells = [0, 1].map((guess) => {
      const share = total ? m[truth][guess] / total : 0;
      const good = truth === guess;
      const colour = good ? `rgba(46, 160, 67, ${0.12 + 0.6 * share})` : `rgba(217, 83, 79, ${0.05 + 0.7 * share})`;
      const cell = el("td", { class: "ev-cm-cell" }, el("div", { class: "ev-cm-count" }, m[truth][guess]), el("small", {}, total ? pct(share) : "-"));
      cell.style.background = colour;
      return cell;
    });
    return el("tr", {}, el("th", {}, `True: ${names[truth]}`), ...cells);
  });
  return el("table", { class: "ev-cm" },
    el("thead", {}, el("tr", {}, el("th", {}), el("th", {}, "Predicted: outside"), el("th", {}, "Predicted: inside"))),
    el("tbody", {}, ...rows));
}

function perClassTable(metrics) {
  const rows = Object.entries(metrics.per_class).map(([name, v]) =>
    el("tr", {}, el("td", {}, name), el("td", {}, pct(v.precision)), el("td", {}, pct(v.recall)), el("td", {}, pct(v.f1)), el("td", {}, v.support)));
  return el("table", {},
    el("thead", {}, el("tr", {}, ...["Class", "Precision", "Recall", "F1", "Images"].map((h) => el("th", {}, h)))),
    el("tbody", {}, ...rows));
}

function scoreHistogram(result) {
  const scored = result.items.filter((item) => item.score !== null && item.score !== undefined);
  if (!scored.length) return el("p", { class: "muted" }, "This model has no score output.");
  const threshold = result.score_kind && result.score_kind.includes("probability") ? 0.5 : 0;
  const values = scored.map((item) => item.score);
  let low = Math.min(...values, threshold), high = Math.max(...values, threshold);
  if (low === high) { low -= 1; high += 1; }
  const bins = 24, step = (high - low) / bins;
  const counts = [Array(bins).fill(0), Array(bins).fill(0)];
  for (const item of scored) counts[item.label][Math.min(bins - 1, Math.floor((item.score - low) / step))] += 1;
  const peak = Math.max(1, ...counts[0], ...counts[1]);
  const W = 640, H = 200, left = 30, bottom = 24, top = 8;
  const plotW = W - left - 8, plotH = H - bottom - top, bw = plotW / bins;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "ev-hist", role: "img", "aria-label": "Score distribution by true class" });
  svg.append(svgEl("line", { x1: left, y1: top + plotH, x2: W - 8, y2: top + plotH, stroke: "#999" }));
  for (let b = 0; b < bins; b += 1) {
    [[0, OUTSIDE_COLOUR], [1, INSIDE_COLOUR]].forEach(([cls, colour], offset) => {
      const h = (counts[cls][b] / peak) * plotH;
      svg.append(svgEl("rect", { x: left + b * bw + offset * (bw / 2), y: top + plotH - h, width: bw / 2 - 1, height: h, fill: colour, opacity: 0.85 }));
    });
  }
  const x = left + ((threshold - low) / (high - low)) * plotW;
  svg.append(svgEl("line", { x1: x, y1: top, x2: x, y2: top + plotH, stroke: "#222", "stroke-dasharray": "4 3" }));
  for (const [text, px, anchor] of [[low.toFixed(2), left, "start"], [threshold.toFixed(2), x, "middle"], [high.toFixed(2), W - 8, "end"]]) {
    const label = svgEl("text", { x: px, y: H - 6, "text-anchor": anchor, "font-size": 11, fill: "#555" });
    label.textContent = text;
    svg.append(label);
  }
  const peakLabel = svgEl("text", { x: 4, y: top + 10, "font-size": 11, fill: "#555" });
  peakLabel.textContent = String(peak);
  svg.append(peakLabel);
  return el("div", {}, svg, el("small", { class: "muted" },
    `Blue = outside images, red = inside images. Dashed line = decision threshold (${result.score_kind || "score"}). Bars on the wrong side of it are mistakes; bars close to it are uncertain.`));
}

function groupTable(result) {
  const rows = result.groups.map((g) => {
    const bar = el("div", { class: "ev-bar" }, el("div", { class: "ev-bar-fill" }));
    bar.firstChild.style.width = pct(g.accuracy);
    bar.firstChild.classList.toggle("bad", g.accuracy < 0.5);
    return el("tr", {}, el("td", {}, g.group_id), el("td", {}, g.class), el("td", {}, `${g.correct}/${g.images}`), el("td", {}, bar, el("small", {}, pct(g.accuracy))));
  });
  return el("table", {},
    el("thead", {}, el("tr", {}, ...["Location group", "True class", "Correct", "Accuracy"].map((h) => el("th", {}, h)))),
    el("tbody", {}, ...rows));
}

function imageUrl(result, item, width) {
  return `/api/evaluate/image?split=${encodeURIComponent(result.split)}&id=${encodeURIComponent(item.image_id)}&w=${width}`;
}

function gallery(result) {
  const PAGE = 24;
  let shown = PAGE;
  const filter = el("select", {},
    el("option", { value: "wrong" }, "All mistakes"),
    el("option", { value: "fp" }, "Outside photos predicted inside"),
    el("option", { value: "fn" }, "Inside photos predicted outside"),
    el("option", { value: "right" }, "Correct predictions"),
    el("option", { value: "all" }, "Everything"));
  const order = el("select", {},
    el("option", { value: "confident" }, "Most confident first"),
    el("option", { value: "uncertain" }, "Least confident first"));
  const grid = el("div", { class: "ev-gallery" });
  const more = el("button", { class: "secondary" }, "Show more");
  const count = el("span", { class: "muted" });

  function matches(item) {
    switch (filter.value) {
      case "wrong": return !item.correct;
      case "fp": return item.label === 0 && item.pred === 1;
      case "fn": return item.label === 1 && item.pred === 0;
      case "right": return item.correct;
      default: return true;
    }
  }

  function draw() {
    const picked = result.items.filter(matches);
    const sign = order.value === "confident" ? -1 : 1;
    picked.sort((a, b) => sign * ((a.margin ?? 0) - (b.margin ?? 0)));
    count.textContent = `${picked.length} image${picked.length === 1 ? "" : "s"}`;
    grid.replaceChildren(...picked.slice(0, shown).map((item) => {
      const truth = item.label === 1 ? "inside" : "outside";
      const guess = item.pred === 1 ? "inside" : "outside";
      const img = el("img", { src: imageUrl(result, item, 480), alt: item.image_id, loading: "lazy" });
      const card = el("figure", { class: `ev-card ${item.correct ? "ok" : "wrong"}` },
        el("a", { href: imageUrl(result, item, 1600), target: "_blank", rel: "noopener" }, img),
        el("figcaption", {},
          el("div", {}, el("strong", {}, item.correct ? "Correct" : "Wrong"), ` | true ${truth}, predicted ${guess}`),
          el("small", { class: "muted" }, `${item.group_id} | ${item.image_id}${item.score === null || item.score === undefined ? "" : ` | score ${item.score.toFixed(2)}`}`)));
      return card;
    }));
    more.classList.toggle("hidden", picked.length <= shown);
  }

  filter.addEventListener("change", () => { shown = PAGE; draw(); });
  order.addEventListener("change", () => { shown = PAGE; draw(); });
  more.addEventListener("click", () => { shown += PAGE; draw(); });
  filter.value = result.items.some((item) => !item.correct) ? "wrong" : "all";
  draw();
  return el("div", {}, el("div", { class: "controls" }, el("label", {}, "Show ", filter), el("label", {}, "Order ", order), count), grid, more);
}

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return `"${text.replace(/"/g, '""')}"`;
}

function exportCsv(result) {
  const header = ["image_id", "group_id", "true_label", "predicted_label", "score", "correct"];
  const rows = result.items.map((i) => [i.image_id, i.group_id, i.label, i.pred, i.score, i.correct].map(csvCell).join(","));
  const blob = new Blob(["\ufeff" + [header.join(","), ...rows].join("\r\n")], { type: "text/csv;charset=utf-8" });
  const link = el("a", { href: URL.createObjectURL(blob), download: `evaluation_${result.split}.csv` });
  link.click();
  URL.revokeObjectURL(link.href);
}

function renderResult(result) {
  const m = result.metrics;
  const nodes = [];
  for (const text of result.warnings) nodes.push(el("div", { class: "warn" }, text));
  if (result.split === "test" && result.test_history) {
    const times = result.test_history.length;
    nodes.push(el("div", { class: times > 1 ? "warn" : "muted" },
      `The test split has been evaluated ${times} time${times === 1 ? "" : "s"} on this version of images.csv. ${times > 1 ? "Repeated evaluation lets the test set influence your decisions; report the first result honestly." : "Report this as your final result."}`));
  }
  nodes.push(el("div", { class: "ev-stats" },
    statCard("Accuracy", pct(m.accuracy), `${m.correct} of ${m.images} correct`),
    statCard("Balanced accuracy", pct(m.balanced_accuracy), "mean recall of both classes"),
    statCard("Macro F1", pct(m.macro_f1), "mean F1 of both classes"),
    statCard("Model", result.model, `${result.split} split | ${result.elapsed_s} s${result.cached ? " | cached" : ""}`)));
  nodes.push(el("section", { class: "card" }, el("h2", {}, "Confusion matrix"), confusionMatrix(m),
    el("p", { class: "muted" }, "Rows are the true class, columns the prediction; percentages are per true class (the diagonal is recall).")));
  nodes.push(el("section", { class: "card" }, el("h2", {}, "Per-class metrics"), perClassTable(m)));
  nodes.push(el("section", { class: "card" }, el("h2", {}, "Score distribution"), scoreHistogram(result)));
  nodes.push(el("section", { class: "card" }, el("h2", {}, "Results by location group"),
    el("p", { class: "muted" }, "Worst groups first. Images of one place are not independent, so a whole group failing matters more than the overall percentage suggests."), groupTable(result)));
  nodes.push(el("section", { class: "card" }, el("h2", {}, "Look at the images"), gallery(result)));
  if (result.failures.length) {
    nodes.push(el("section", { class: "card" }, el("h2", {}, "Images that could not be classified"),
      el("ul", {}, ...result.failures.map((f) => el("li", {}, `${f.image_id} (${f.group_id}): ${f.error}`)))));
  }
  const exportButton = el("button", { class: "secondary" }, "Export per-image results (CSV)");
  exportButton.addEventListener("click", () => exportCsv(result));
  nodes.push(el("div", { class: "controls" }, exportButton));
  return nodes;
}

function initEvaluatePage() {
  const split = document.getElementById("evSplit");
  const run = document.getElementById("evRun");
  const rerun = document.getElementById("evRefresh");
  const gate = document.getElementById("evTestGate");
  const confirmBox = document.getElementById("evTestConfirm");
  const status = document.getElementById("evStatus");
  const output = document.getElementById("evOutput");
  let busy = false;

  function sync() {
    const isTest = split.value === "test";
    gate.classList.toggle("hidden", !isTest);
    run.disabled = busy || (isTest && !confirmBox.checked);
    rerun.disabled = busy || output.childElementCount === 0 || (isTest && !confirmBox.checked);
  }

  async function execute(refresh) {
    busy = true;
    sync();
    status.textContent = "Evaluating... every image in the split is decoded once.";
    try {
      const result = await api("/api/evaluate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ split: split.value, confirm_test: split.value === "test" && confirmBox.checked, refresh }),
      });
      output.replaceChildren(...renderResult(result));
      status.textContent = `Done: ${result.metrics.images} images.`;
    } catch (error) {
      output.replaceChildren();
      status.textContent = error.message;
    } finally {
      busy = false;
      sync();
      refreshBadge();
    }
  }

  split.addEventListener("change", () => { output.replaceChildren(); status.textContent = ""; sync(); });
  confirmBox.addEventListener("change", sync);
  run.addEventListener("click", () => execute(false));
  rerun.addEventListener("click", () => execute(true));
  sync();
}

document.addEventListener("DOMContentLoaded", () => {
  if (document.body.dataset.page === "evaluate") initEvaluatePage();
});
