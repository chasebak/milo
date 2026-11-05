// Restock Explorer: a viewer for docs/data/*.json.
// Why this file computes nothing: every number is written by
// `python -m experiments.viz build` from the tested experiment functions and checked
// in tests/test_viz.py. This script only looks values up by budget and policy
// and formats them (plus placing precomputed 0-1 positions on the SVG axis),
// so the page cannot drift from the README.
"use strict";

(function () {
  var state = { grid: null, method: null, saa: null, index: 0, buyPolicy: "optimizer", chart: null, saaChart: null };

  function byId(id) { return document.getElementById(id); }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ---- formatting only -------------------------------------------------
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function grouped(v, digits) {
    return Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }
  function money(v, digits) {
    if (!isNum(v)) return "n/a";
    var d = digits === undefined ? 2 : digits;
    var text = "$" + grouped(v, d);
    if (v < 0 && text !== "$" + grouped(0, d)) return "-" + text;
    return text;
  }
  function money0(v) { return money(v, 0); }
  function plain(v, digits) {
    if (!isNum(v)) return "n/a";
    var text = grouped(v, digits);
    return v < 0 ? "-" + text : text;
  }
  function ratio(v) { return isNum(v) ? v.toFixed(3) : "n/a"; }
  function pct(v) { return isNum(v) ? v.toFixed(2) + "%" : "n/a"; }
  function signedPct1(v) {
    if (!isNum(v)) return "n/a";
    return (v > 0 ? "+" : "") + v.toFixed(1) + "%";
  }
  function seconds(v, digits) { return isNum(v) ? v.toFixed(digits) + " s" : "n/a"; }
  function count(v) { return isNum(v) ? String(v) : "n/a"; }
  function shortMoney(v) { return v >= 1000 ? "$" + v / 1000 + "k" : "$" + v; }
  function budgetList(list) {
    if (!list.length) return "no solved budget";
    var parts = [];
    for (var i = 0; i < list.length; i++) parts.push(money0(list[i]));
    return parts.join(", ");
  }

  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    if (attrs) {
      var keys = Object.keys(attrs);
      for (var i = 0; i < keys.length; i++) node.setAttribute(keys[i], attrs[keys[i]]);
    }
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function svgEl(tag, attrs, text) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    var keys = Object.keys(attrs);
    for (var i = 0; i < keys.length; i++) node.setAttribute(keys[i], attrs[keys[i]]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // ---- lookups -----------------------------------------------------------
  function budgets() { return state.grid.budgets; }
  function budget() { return budgets()[state.index]; }
  function cellAt(index) { return state.grid.cells[index]; }
  function cell() { return cellAt(state.index); }
  function policies() { return state.method.policies; }
  function capital(text) { return text.charAt(0).toUpperCase() + text.slice(1); }
  function label(policy) { return capital(state.method.policy_labels[policy]); }
  function shortLabel(policy) { return state.method.policy_short_labels[policy]; }
  function product(id) {
    var list = state.grid.products;
    for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
    return null;
  }
  function indexOfBudget(value) {
    var list = budgets();
    for (var i = 0; i < list.length; i++) if (list[i] === value) return i;
    return -1;
  }
  function gapLabel() { return (state.method.settings.mip_gap * 100).toFixed(2) + "%"; }

  // ---- static text -------------------------------------------------------
  function fillWhyBoxes() {
    var boxes = document.querySelectorAll(".why[data-why]");
    for (var i = 0; i < boxes.length; i++) {
      var entry = state.method.why[boxes[i].getAttribute("data-why")];
      boxes[i].textContent = "";
      boxes[i].appendChild(el("b", null, "Why this metric"));
      boxes[i].appendChild(el("p", null, entry.question));
      boxes[i].appendChild(el("p", null, "Kind: " + entry.kind));
      boxes[i].appendChild(el("p", null, "Could lie if: " + entry.lie));
    }
  }
  function fillProvenance() {
    var pills = document.querySelectorAll(".provenance[data-prov]");
    for (var i = 0; i < pills.length; i++) {
      pills[i].textContent = state.method.provenance[pills[i].getAttribute("data-prov")];
    }
  }
  function fillStatic() {
    var m = state.method;
    byId("problem-label").textContent = m.label + " " + capital(m.provenance.page) + ".";
    byId("n-eval").textContent = plain(m.settings.n_eval, 0);
    byId("n-train").textContent = count(m.settings.n_scenarios);
    byId("footer-text").textContent = m.footer;
    var f = m.grid_facts;
    var wins = f.greedy_rord_out_wins;
    byId("curve-note").textContent =
      "Widest optimizer lead over the greedy list: " + pct(f.widest_lead_in.pct) + " at " + money0(f.widest_lead_in.budget) +
      " in-sample, " + pct(f.widest_lead_out.pct) + " at " + money0(f.widest_lead_out.budget) + " out-of-sample. " +
      "The greedy list earns more per restock dollar out-of-sample at " + budgetList(wins) +
      "; it earns more total profit out-of-sample at " + budgetList(f.greedy_profit_out_wins) + ". " +
      f.n_solves + " MILP solves, " + seconds(f.solve_seconds_min, 1) + " to " + seconds(f.solve_seconds_max, 1) +
      " each, " + (f.all_solves_optimal ? "all optimal within a " + gapLabel() + " gap." : "not all optimal within the gap.");
  }

  // ---- slider ------------------------------------------------------------
  function buildSlider() {
    var slider = byId("budget");
    var list = budgets();
    slider.max = String(list.length - 1);
    var ticks = byId("ticks");
    ticks.textContent = "";
    var labelled = state.method.tick_budgets;
    for (var i = 0; i < list.length; i++) {
      var stop = el("span", { "class": "stop", "data-index": String(i) });
      if (i === 0) stop.classList.add("first");
      if (i === list.length - 1) stop.classList.add("last");
      stop.style.left = (i / (list.length - 1)) * 100 + "%";
      stop.appendChild(el("span", { "class": "mark" }));
      if (labelled.indexOf(list[i]) !== -1) stop.appendChild(document.createTextNode(shortMoney(list[i])));
      ticks.appendChild(stop);
    }
    slider.addEventListener("input", function () { select(Number(slider.value)); });
  }
  function select(index) {
    state.index = index;
    var slider = byId("budget");
    slider.value = String(index);
    slider.setAttribute("aria-valuetext", money0(budget()));
    byId("budget-now").textContent = money0(budget());
    var spans = document.querySelectorAll("[data-budget-now]");
    for (var i = 0; i < spans.length; i++) spans[i].textContent = money0(budget());
    var stops = document.querySelectorAll("#ticks .stop");
    for (var j = 0; j < stops.length; j++) stops[j].classList.toggle("on", j === index);
    try { history.replaceState(null, "", "#budget=" + budget()); } catch (e) { /* file:// */ }
    renderTiles();
    renderBuyList();
    markFallbackRow();
    if (state.chart) state.chart.update("none");
  }

  // ---- tiles -------------------------------------------------------------
  function tile(className, labelText, valueText, noteText) {
    var t = el("div", { "class": "tile " + className });
    t.appendChild(el("div", { "class": "label" }, labelText));
    t.appendChild(el("div", { "class": "value" }, valueText));
    t.appendChild(el("div", { "class": "note" }, noteText));
    return t;
  }
  function policyNote(p) {
    return "in-sample " + ratio(p.rord_in) + " · out-of-sample profit " + money(p.out_of_sample) +
      " · spent " + money(p.spend) + " of " + money0(budget());
  }
  function renderTiles() {
    var c = cell();
    var o = c.policies.optimizer, g = c.policies.greedy, t = c.policies.today_prices;
    var box = byId("tiles");
    box.textContent = "";
    box.appendChild(tile("headline", label("optimizer"), ratio(o.rord_out), policyNote(o)));
    box.appendChild(tile("baseline", label("greedy"), ratio(g.rord_out), policyNote(g)));
    box.appendChild(tile("baseline", label("today_prices") + " (fair baseline)", ratio(t.rord_out), policyNote(t)));
    box.appendChild(tile("", "Optimizer lead over the greedy list, out-of-sample profit",
      money(c.lead_vs_greedy_out), pct(c.lead_vs_greedy_out_pct) + " out-of-sample · " + pct(c.lead_vs_greedy_in_pct) +
      " in-sample · over today's prices " + money(c.lead_vs_today_out)));
    box.appendChild(tile("", "Optimizer's-curse gap (in-sample − out-of-sample)", money(o.curse_gap),
      "in-sample " + money(o.in_sample) + " · out-of-sample " + money(o.out_of_sample) +
      " (p10 " + money(o.out_of_sample_p10) + ", p90 " + money(o.out_of_sample_p90) + ")"));
    var na = budget() === 0 ? " At $0 nothing is spent, so profit per restock dollar is n/a." : "";
    byId("tiles-note").textContent =
      "Profit per restock dollar = (expected profit at this budget − the same policy's $0 plan) ÷ what it spent." + na +
      " The optimizer's solve is within " + money(c.tolerance) + " of the best plan (mip_gap × |bound|).";
  }

  // ---- buy list ----------------------------------------------------------
  function buildBuyButtons() {
    var group = byId("buy-policies");
    var list = policies();
    for (var i = 0; i < list.length; i++) {
      var button = el("button", { type: "button", "data-policy": list[i], "aria-pressed": String(list[i] === state.buyPolicy) },
        shortLabel(list[i]));
      button.title = label(list[i]);
      button.addEventListener("click", onBuyButton);
      group.appendChild(button);
    }
  }
  function onBuyButton(event) {
    state.buyPolicy = event.currentTarget.getAttribute("data-policy");
    var buttons = byId("buy-policies").querySelectorAll("button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-pressed", String(buttons[i].getAttribute("data-policy") === state.buyPolicy));
    }
    renderBuyList();
  }
  function headerRow(table, names, numericFrom) {
    var thead = el("thead");
    var tr = el("tr");
    for (var i = 0; i < names.length; i++) {
      tr.appendChild(el("th", i >= numericFrom ? { "class": "num", scope: "col" } : { scope: "col" }, names[i]));
    }
    thead.appendChild(tr);
    table.appendChild(thead);
  }
  function row(values, numericFrom, rowClass) {
    var tr = el("tr", rowClass ? { "class": rowClass } : null);
    for (var i = 0; i < values.length; i++) {
      tr.appendChild(el("td", i >= numericFrom ? { "class": "num" } : null, values[i]));
    }
    return tr;
  }
  function edgeNote(s) {
    if (!s.price_edits) return "";
    var high = plain(state.method.settings.price_grid_high, 2);
    var low = plain(state.method.settings.price_grid_low, 2);
    var text = " " + count(s.price_edits_at_top_edge) + " of the " + count(s.price_edits) + " moved prices sit at the " +
      high + "× grid edge (flag M2)";
    if (s.price_edits_at_top_edge > 0) {
      return text + ": the grid, not the data, binds, so these moves measure the " + low + "–" + high + "× grid as much as demand.";
    }
    return text + ".";
  }
  function renderBuyList() {
    var policy = state.buyPolicy;
    var list = cell().buy_lists[policy];
    var s = list.summary;
    byId("buy-summary").textContent =
      label(policy) + " buys " + count(s.units_bought) + " units of " + count(s.products_bought) + " products for " +
      money(s.spend) + " of the " + money0(budget()) + " budget, and moves " + count(s.price_edits) +
      " stocked products off today's price. " + count(s.stocked_at_today_price) +
      " other stocked products keep today's price and are not listed." + edgeNote(s);
    var table = byId("buy-table");
    table.textContent = "";
    headerRow(table, ["product", "on hand", "buy", "spend", "price", "today", "change"], 1);
    var tbody = el("tbody");
    for (var i = 0; i < list.rows.length; i++) {
      var r = list.rows[i];
      var p = product(r.product);
      var buyText = r.buy > 0 ? count(r.buy) : "reprice only";
      tbody.appendChild(row([p.label, plain(p.on_hand, 0), buyText, money(r.spend), money(r.price), money(p.price_now),
        r.price_moved ? signedPct1(r.price_change_pct) : "same"], 1, ""));
    }
    if (!list.rows.length) tbody.appendChild(row(["nothing bought or repriced at this budget", "", "", "", "", "", ""], 1, ""));
    table.appendChild(tbody);
  }

  // ---- fallback table ----------------------------------------------------
  function renderFallbackTable() {
    var table = byId("fallback-table");
    table.textContent = "";
    var names = ["budget"];
    var list = policies();
    for (var i = 0; i < list.length; i++) names.push(shortLabel(list[i]) + ": out-of-sample profit");
    names.push("Optimizer: in-sample");
    names.push("Optimizer: p10");
    names.push("Optimizer: p90");
    for (var k = 0; k < list.length; k++) names.push(shortLabel(list[k]) + ": profit per $ (out)");
    headerRow(table, names, 1);
    var tbody = el("tbody");
    var all = budgets();
    for (var j = 0; j < all.length; j++) {
      var c = cellAt(j);
      var values = [money0(all[j])];
      for (var a = 0; a < list.length; a++) values.push(money(c.policies[list[a]].out_of_sample));
      values.push(money(c.policies.optimizer.in_sample));
      values.push(money(c.policies.optimizer.out_of_sample_p10));
      values.push(money(c.policies.optimizer.out_of_sample_p90));
      for (var b = 0; b < list.length; b++) values.push(ratio(c.policies[list[b]].rord_out));
      tbody.appendChild(row(values, 1, ""));
    }
    table.appendChild(tbody);
  }
  function markFallbackRow() {
    var rows = byId("fallback-table").querySelectorAll("tbody tr");
    for (var i = 0; i < rows.length; i++) rows[i].classList.toggle("chosen", i === state.index);
  }

  // ---- profit curve ------------------------------------------------------
  // Inline cursor (~20 lines, not a library plugin): a vertical line at the slider budget.
  var cursorPlugin = {
    id: "budgetCursor",
    afterDatasetsDraw: function (chart) {
      var x = chart.scales.x.getPixelForValue(budget());
      var area = chart.chartArea;
      var ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = cssVar("--text-muted");
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, area.top);
      ctx.lineTo(x, area.bottom);
      ctx.stroke();
      ctx.restore();
    }
  };
  function series(policy, key) {
    var points = [];
    var all = budgets();
    for (var j = 0; j < all.length; j++) points.push({ x: all[j], y: cellAt(j).policies[policy][key] });
    return points;
  }
  function lineSet(text, points, color, dash, width) {
    return { label: text, data: points, borderColor: color, backgroundColor: color, borderDash: dash, borderWidth: width,
      pointRadius: 1.5, pointHoverRadius: 4, tension: 0, fill: false };
  }
  function axisOptions(text, grid, title) {
    return { title: { display: true, text: title, color: text }, ticks: { color: text }, grid: { color: grid } };
  }
  function buildChart() {
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    var accent = cssVar("--accent");
    var band = cssVar("--band");
    var p10 = lineSet("Optimizer p10", series("optimizer", "out_of_sample_p10"), band, [], 0);
    p10.pointRadius = 0;
    var p90 = lineSet("Optimizer p10–p90, out-of-sample", series("optimizer", "out_of_sample_p90"), band, [], 0);
    p90.pointRadius = 0;
    p90.fill = "-1";
    var datasets = [
      p10, p90,
      lineSet("Optimizer, out-of-sample", series("optimizer", "out_of_sample"), accent, [], 3),
      lineSet("Optimizer, in-sample", series("optimizer", "in_sample"), accent, [6, 4], 1.5),
      lineSet("Greedy list, out-of-sample", series("greedy", "out_of_sample"), cssVar("--series-2"), [], 2),
      lineSet("Today's prices, out-of-sample", series("today_prices", "out_of_sample"), cssVar("--baseline"), [2, 3], 2)
    ];
    var text = cssVar("--text-muted");
    var grid = cssVar("--border");
    var tickBudgets = state.method.tick_budgets;
    var all = budgets();
    var x = axisOptions(text, grid, "restock budget");
    x.type = "linear";
    x.min = all[0];
    x.max = all[all.length - 1];
    x.ticks.callback = function (v) { return shortMoney(v); };
    x.afterBuildTicks = function (axis) {
      var ticks = [];
      for (var k = 0; k < tickBudgets.length; k++) ticks.push({ value: tickBudgets[k] });
      axis.ticks = ticks;
    };
    var y = axisOptions(text, grid, "expected profit ($)");
    y.ticks.callback = function (v) { return money0(v); };
    state.chart = new window.Chart(byId("chart"), {
      type: "line",
      data: { datasets: datasets },
      plugins: [cursorPlugin],
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        scales: { x: x, y: y },
        plugins: {
          legend: {
            labels: { color: cssVar("--text"), boxWidth: 14, filter: function (item) { return item.datasetIndex !== 0; } }
          },
          tooltip: {
            filter: function (item) { return item.datasetIndex > 1; },
            callbacks: {
              title: function (items) { return items.length ? money0(items[0].parsed.x) + " budget" : ""; },
              label: function (item) { return item.dataset.label + ": " + money(item.parsed.y); }
            }
          }
        },
        onClick: function (event, elements) {
          if (elements.length) select(elements[0].index);  // points are in budget order
        }
      }
    });
  }

  // ---- EEV -> RP -> WS bracket (inline SVG) ------------------------------
  var AXIS = { left: 20, right: 340, y: 78 };
  function xAt(position) { return AXIS.left + position * (AXIS.right - AXIS.left); }
  function brace(group, from, to, cls, text) {
    var x1 = xAt(from), x2 = xAt(to);
    group.appendChild(svgEl("path", { "class": "brace " + "brace-" + cls,
      d: "M" + x1 + " 40 L" + x1 + " 32 L" + x2 + " 32 L" + x2 + " 40" }));
    group.appendChild(svgEl("text", { "class": "brace-label " + cls, x: String((x1 + x2) / 2), y: "25", "text-anchor": "middle" }, text));
  }
  function renderBracket() {
    var b = state.method.bracket;
    var ticks = byId("bracket-ticks");
    var dots = byId("bracket-dots");
    var braces = byId("bracket-braces");
    ticks.textContent = "";
    dots.textContent = "";
    braces.textContent = "";
    for (var i = 0; i < b.ticks.length; i++) {
      var tx = xAt(b.ticks[i].position);
      ticks.appendChild(svgEl("line", { "class": "tick", x1: String(tx), y1: String(AXIS.y), x2: String(tx), y2: String(AXIS.y + 4) }));
      ticks.appendChild(svgEl("text", { "class": "tick-label", x: String(tx), y: String(AXIS.y + 16), "text-anchor": "middle" },
        plain(b.ticks[i].value, 0)));
    }
    brace(braces, b.positions.eev, b.positions.rp, "vss", "VSS " + plain(b.vss, 2));
    brace(braces, b.positions.rp, b.positions.ws, "evpi", "EVPI " + plain(b.evpi, 2));
    var names = [["eev", "EEV"], ["rp", "RP"], ["ws", "WS"]];
    for (var j = 0; j < names.length; j++) {
      var dx = xAt(b.positions[names[j][0]]);
      dots.appendChild(svgEl("circle", { "class": "dot-" + names[j][0], cx: String(dx), cy: String(AXIS.y), r: "5" }));
      dots.appendChild(svgEl("text", { "class": "dot-label", x: String(dx), y: String(AXIS.y - 12), "text-anchor": "middle" },
        names[j][1] + " " + plain(b[names[j][0]], 0)));
    }
    byId("bracket-vss").textContent = "VSS = RP − EEV = " + money(b.vss) + " (" + pct(b.vss_pct_of_rp) +
      " of RP): what planning on the average scenario, instead of the whole spread, would cost.";
    byId("bracket-evpi").textContent = "EVPI = WS − RP = " + money(b.evpi) + " (" + pct(b.evpi_pct_of_rp) +
      " of RP): the most that perfect demand forecasts could add. Both at " + money0(b.budget) + ", in-sample.";
    var table = byId("bracket-table");
    table.textContent = "";
    headerRow(table, ["quantity", "expected profit"], 1);
    var tbody = el("tbody");
    tbody.appendChild(row(["EEV (average-scenario plan, re-valued on the scenarios)", money(b.eev)], 1, ""));
    tbody.appendChild(row(["RP (two-stage stochastic optimum)", money(b.rp)], 1, "chosen"));
    tbody.appendChild(row(["WS (wait-and-see bound)", money(b.ws)], 1, ""));
    tbody.appendChild(row(["VSS = RP − EEV", money(b.vss) + " (" + pct(b.vss_pct_of_rp) + ")"], 1, ""));
    tbody.appendChild(row(["EVPI = WS − RP", money(b.evpi) + " (" + pct(b.evpi_pct_of_rp) + ")"], 1, ""));
    table.appendChild(tbody);
  }

  // ---- SAA and solve time ------------------------------------------------
  function renderSaaTable() {
    var table = byId("saa-table");
    table.textContent = "";
    headerRow(table, ["N (training scenarios)", "in-sample mean", "in-sample std", "out-of-sample mean", "out-of-sample std",
      "curse gap", "solve (mean)"], 1);
    var tbody = el("tbody");
    var rows = state.saa.rows;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      tbody.appendChild(row([count(r.n_scenarios), money(r.in_sample_mean), plain(r.in_sample_std, 2), money(r.out_of_sample_mean),
        plain(r.out_of_sample_std, 2), money(r.curse_gap), seconds(r.solve_seconds_mean, 1)], 1, ""));
    }
    var pin = state.saa.pinned;
    tbody.appendChild(row([count(pin.n_scenarios) + " (pinned seed " + pin.seed + ")", money(pin.in_sample), "", money(pin.out_of_sample),
      "", money(pin.curse_gap), ""], 1, "chosen"));
    table.appendChild(tbody);
    var first = rows[0], last = rows[rows.length - 1];
    var spread = state.saa.seed_spread;
    var shape = state.saa.curse_gap_shape;
    var path = "";
    if (!shape.monotone) {
      var rises = [];
      for (var k = 0; k < shape.rises.length; k++) {
        var rise = shape.rises[k];
        rises.push("N = " + rise.to_n + " (" + money(rise.to_gap) + ") is above N = " + rise.from_n + " (" + money(rise.from_gap) + ")");
      }
      path = ", but not smoothly: " + rises.join("; ");
    }
    byId("saa-note").textContent =
      "The mean curse gap shrinks from " + money(first.curse_gap) + " at N = " + first.n_scenarios + " to " + money(last.curse_gap) +
      " at N = " + last.n_scenarios + path + ". The page's pinned problem (N = " + pin.n_scenarios + ", seed " + pin.seed + ") has a gap of " +
      money(pin.curse_gap) + ": its in-sample RP " + money(pin.in_sample) + " sits " + money(pin.minus_seed_mean) + " above the " +
      pin.n_seeds + "-seed mean " + money(spread.mean) + " (" + plain(pin.in_seed_sds, 2) + " sds, rank " + pin.rank_from_top +
      " of " + pin.n_seeds + "), a mild draw; the out-of-sample tiles are still the ones to read.";
  }
  function buildSaaChart() {
    if (state.saaChart) { state.saaChart.destroy(); state.saaChart = null; }
    var rows = state.saa.rows;
    var inPoints = [], outPoints = [];
    for (var i = 0; i < rows.length; i++) {
      inPoints.push({ x: rows[i].n_scenarios, y: rows[i].in_sample_mean });
      outPoints.push({ x: rows[i].n_scenarios, y: rows[i].out_of_sample_mean });
    }
    var pin = state.saa.pinned;
    var accent = cssVar("--accent");
    var other = cssVar("--series-3");
    var text = cssVar("--text-muted");
    var grid = cssVar("--border");
    var seeds = " (" + rows[0].n_seeds + " seeds)";
    var inSet = lineSet("In-sample mean" + seeds, inPoints, accent, [6, 4], 2);
    inSet.showLine = true;
    inSet.pointRadius = 3;
    var outSet = lineSet("Out-of-sample mean" + seeds, outPoints, other, [], 2);
    outSet.showLine = true;
    outSet.pointRadius = 3;
    var pinIn = lineSet("Pinned seed, in-sample", [{ x: pin.n_scenarios, y: pin.in_sample }], accent, [], 1);
    pinIn.pointStyle = "triangle";
    pinIn.pointRadius = 6;
    var pinOut = lineSet("Pinned seed, out-of-sample", [{ x: pin.n_scenarios, y: pin.out_of_sample }], other, [], 1);
    pinOut.pointStyle = "triangle";
    pinOut.pointRadius = 6;
    var x = axisOptions(text, grid, "training scenarios N (log scale)");
    x.type = "logarithmic";
    x.afterBuildTicks = function (axis) {
      var ticks = [];
      for (var k = 0; k < rows.length; k++) ticks.push({ value: rows[k].n_scenarios });
      axis.ticks = ticks;
    };
    x.ticks.callback = function (v) { return String(v); };
    var y = axisOptions(text, grid, "expected profit ($)");
    y.ticks.callback = function (v) { return money0(v); };
    state.saaChart = new window.Chart(byId("saa-chart"), {
      type: "scatter",
      data: { datasets: [inSet, outSet, pinIn, pinOut] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: { x: x, y: y },
        plugins: {
          legend: { labels: { color: cssVar("--text"), boxWidth: 14 } },
          tooltip: { callbacks: { label: function (item) { return item.dataset.label + ", N = " + item.parsed.x + ": " + money(item.parsed.y); } } }
        }
      }
    });
  }
  function renderHullTable() {
    var table = byId("hull-table");
    table.textContent = "";
    headerRow(table, ["budget", "transform", "incumbent", "LP bound", "build", "solve", "result"], 2);
    var tbody = el("tbody");
    var rows = state.method.hull_bigm;
    var limit = state.method.settings.time_limit;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var result = r.optimal ? "optimal within a " + gapLabel() + " gap" : "time limit (" + limit + " s)";
      tbody.appendChild(row([money0(r.budget), r.transform, money(r.objective), money(r.lp_relaxation_objective),
        seconds(r.build_seconds, 2), seconds(r.solve_seconds, 1), result], 2, r.transform === "hull" ? "chosen" : ""));
    }
    table.appendChild(tbody);
  }

  function showFallback(boxId, noteId, detailsId) {
    byId(boxId).hidden = true;
    byId(noteId).hidden = false;
    byId(detailsId).open = true;
  }
  function buildCharts() {
    buildChart();
    buildSaaChart();
  }

  // ---- start -------------------------------------------------------------
  function start(grid, method, saa) {
    state.grid = grid;
    state.method = method;
    state.saa = saa;
    fillWhyBoxes();
    fillProvenance();
    fillStatic();
    buildSlider();
    buildBuyButtons();
    renderFallbackTable();
    renderBracket();
    renderSaaTable();
    renderHullTable();
    var wanted = indexOfBudget(method.default_budget);
    var match = /budget=(\d+)/.exec(location.hash);
    if (match && indexOfBudget(Number(match[1])) !== -1) wanted = indexOfBudget(Number(match[1]));
    if (window.Chart) {
      buildCharts();
      var media = window.matchMedia("(prefers-color-scheme: dark)");
      if (media.addEventListener) media.addEventListener("change", buildCharts); else media.addListener(buildCharts);
    } else {
      showFallback("chart-box", "chart-fallback", "fallback-details");
      showFallback("saa-box", "saa-fallback", "saa-details");
    }
    select(wanted);
    // Why: a shared link or the back button changes only the hash, which
    // does not reload the page; follow it so the URL and the slider agree.
    window.addEventListener("hashchange", function () {
      var changed = /budget=(\d+)/.exec(location.hash);
      if (!changed) return;
      var index = indexOfBudget(Number(changed[1]));
      if (index !== -1 && index !== state.index) select(index);
    });
  }
  function getJson(path) {
    return fetch(path).then(function (response) {
      if (!response.ok) throw new Error(path + " " + response.status);
      return response.json();
    });
  }
  Promise.all([getJson("data/budget_grid.json"), getJson("data/method.json"), getJson("data/saa.json")])
    .then(function (parts) { start(parts[0], parts[1], parts[2]); })
    .catch(function () { byId("load-error").hidden = false; });
})();
