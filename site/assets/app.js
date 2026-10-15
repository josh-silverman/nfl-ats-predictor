/* Josh's NFL Elo: renders site/data/site-data.js (window.SITE_DATA). No framework.
   All text goes in via textContent, never innerHTML. */
(function () {
  "use strict";
  const D = window.SITE_DATA;
  const COLORS = { ours: "#3987e5", vegas: "#d95926", pick: ["#3987e5", "#d95926", "#199e70"], grid: "#253043", text: "#c3c2b7" };

  // ---------- tiny helpers ----------
  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "style") Object.assign(n.style, v);
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false)
      n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    return n;
  }
  const pct = (x) => Math.round(x * 100) + "%";
  const team = (t) => D.teams[t] || { name: t, color: "#888" };
  const chip = (t) => el("span", { class: "chip", style: { background: team(t).color }, "aria-hidden": "true" });
  const teamLabel = (t) => el("span", { class: "team", title: team(t).name }, chip(t), " ", t);
  const term = (word, tip) => el("span", { class: "term", "data-tip": tip, tabindex: "0" }, word);

  const TIPS = {
    elo: "A rating for each team, built only from final scores. Win and it goes up; lose and it goes down. Beating a strong team moves it more.",
    line: "Our own point spread from the ratings: how many points we expect the favorite to win by.",
    vegas: "The betting market's chance to win, worked out from the betting odds when we locked our picks.",
    locked: "Saved and committed to GitHub before the first game of the week kicked off, so it can't be changed after the fact.",
    backfill: "Weeks 1–5 were predicted after they were played (same rules, no peeking at results), so they don't count in the record.",
    gap: "We and Vegas differ by 15+ points. That usually means Vegas knows something our model doesn't, often a quarterback injury.",
  };

  // ---------- tooltips (hover, keyboard focus, or tap) ----------
  const tip = document.getElementById("tooltip");
  function showTip(target) {
    tip.textContent = target.getAttribute("data-tip");
    tip.hidden = false;
    const r = target.getBoundingClientRect();
    const left = Math.min(Math.max(8, r.left), window.innerWidth - tip.offsetWidth - 8);
    const top = r.bottom + 6 + tip.offsetHeight > window.innerHeight ? r.top - tip.offsetHeight - 6 : r.bottom + 6;
    tip.style.left = left + "px"; tip.style.top = top + "px";
  }
  const hideTip = () => { tip.hidden = true; };
  document.addEventListener("pointerover", (e) => { const t = e.target.closest("[data-tip]"); if (t) showTip(t); });
  document.addEventListener("pointerout", (e) => { if (e.target.closest("[data-tip]")) hideTip(); });
  document.addEventListener("focusin", (e) => { const t = e.target.closest("[data-tip]"); if (t) showTip(t); });
  document.addEventListener("focusout", hideTip);
  document.addEventListener("click", (e) => { const t = e.target.closest("[data-tip]"); t ? showTip(t) : hideTip(); });
  window.addEventListener("scroll", hideTip, { passive: true });

  // ---------- tabs (state lives in the URL hash so links are shareable) ----------
  const TABS = ["week", "record", "rankings", "how"];
  const rendered = {};
  function openTab(name) {
    if (!TABS.includes(name)) name = "week";
    for (const t of TABS) {
      document.getElementById("tab-" + t).hidden = t !== name;
      document.querySelector(`[data-tab="${t}"]`).setAttribute("aria-selected", String(t === name));
    }
    if (!rendered[name]) { RENDER[name](document.getElementById("tab-" + name)); rendered[name] = true; }
  }
  document.querySelectorAll("[data-tab]").forEach((b) =>
    b.addEventListener("click", () => { location.hash = b.dataset.tab; }));
  window.addEventListener("hashchange", () => openTab(location.hash.slice(1)));

  // ---------- This Week ----------
  function gameCard(g) {
    const pHome = g.p_home, pAway = 1 - pHome;
    const fav = pHome >= 0.5 ? g.home : g.away;
    const bar = el("div", { class: "bar", role: "img",
      "aria-label": `${g.away} ${pct(pAway)}, ${g.home} ${pct(pHome)}` + (g.p_vegas_home !== null ? `; Vegas ${g.home} ${pct(g.p_vegas_home)}` : "") },
      el("div", { class: "seg" + (fav === g.away ? "" : " dim"), style: { width: `calc(${pAway * 100}% - 1px)`, background: team(g.away).color } }),
      el("div", { class: "seg" + (fav === g.home ? "" : " dim"), style: { width: `calc(${pHome * 100}% - 1px)`, background: team(g.home).color } }));
    let vegasText = null;
    if (g.p_vegas_home !== null) {
      bar.append(el("div", { class: "vegas-mark", style: { left: `calc(${(1 - g.p_vegas_home) * 100}% - 1px)` }, title: "Vegas" }));
      const vFav = g.p_vegas_home >= 0.5 ? g.home : g.away;
      vegasText = el("span", {}, term("Vegas", TIPS.vegas), `: ${vFav} ${pct(Math.max(g.p_vegas_home, 1 - g.p_vegas_home))}`);
    }
    let result = null;
    if (g.result) {
      const r = g.result;
      const verdict = r.winner === null ? el("span", {}, "Tie")
        : r.we_were_right ? el("span", { class: "right" }, "✓ Picked right") : el("span", { class: "wrong" }, "✗ Missed");
      result = el("div", { class: "result" }, `Final: ${g.away} ${r.away_score}, ${g.home} ${r.home_score} · `, verdict);
    }
    return el("article", { class: "card" },
      el("div", { class: "card-top" },
        el("div", { class: "matchup" }, teamLabel(g.away), el("span", { class: "at" }, "@"), teamLabel(g.home)),
        el("div", { class: "kick" }, g.kickoff)),
      el("div", { class: "bar-labels" },
        el("span", { class: fav === g.away ? "fav" : "" }, `${g.away} ${pct(pAway)}`),
        el("span", { class: fav === g.home ? "fav" : "" }, `${g.home} ${pct(pHome)}`)),
      bar,
      el("div", { class: "card-meta" },
        el("span", {}, term("Our line", TIPS.line), `: ${g.our_line}`),
        vegasText,
        g.neutral ? el("span", { class: "tag" }, "Neutral site") : null,
        g.big_gap ? el("span", { class: "tag gap term", "data-tip": TIPS.gap, tabindex: "0" }, "Big gap with Vegas") : null),
      result);
  }

  function renderWeek(root) {
    const weeks = D.weeks;
    if (!weeks.length) { root.append(el("p", {}, "No predictions yet.")); return; }
    const select = el("select", { "aria-label": "Choose a week" },
      weeks.map((w) => el("option", { value: w.week, selected: w.week === D.meta.current_week },
        `Week ${w.week}` + (w.mode === "backfill" ? " (backfill)" : ""))));
    const status = el("div", {});
    const summary = el("p", { class: "week-summary" });
    const cards = el("div", { class: "cards" });
    function draw() {
      const w = weeks.find((x) => x.week === Number(select.value));
      status.replaceChildren(
        w.mode === "live" && w.lock
          ? (w.lock.locked
            ? el("a", { class: "badge " + (w.lock.before_kickoff ? "ok" : "warn"), href: w.lock.url, target: "_blank", rel: "noopener",
                "data-tip": TIPS.locked }, "🔒 " + w.lock.text)
            : el("span", { class: "badge warn" }, "Not locked yet"))
          : el("div", { class: "banner" }, "Backfill: these picks were made after the games were played, using only earlier results. They don't count in the record. ",
              term("What's backfill?", TIPS.backfill)));
      const done = w.games.filter((g) => g.result && g.result.winner !== null);
      if (done.length) {
        const ours = done.filter((g) => g.result.we_were_right).length;
        const vegasDone = done.filter((g) => g.result.vegas_was_right !== null);
        const vegas = vegasDone.filter((g) => g.result.vegas_was_right).length;
        summary.textContent = `We picked ${ours} of ${done.length} right this week · Vegas ${vegas} of ${vegasDone.length}`;
      } else summary.textContent = `${w.games.length} games · bars show each team's chance to win`;
      cards.replaceChildren(...w.games.map(gameCard));
    }
    select.addEventListener("change", draw);
    root.append(
      el("h2", {}, "This week's games"),
      el("p", { class: "sub" }, "Each bar shows each team's chance to win from our ", term("Elo", TIPS.elo),
        " ratings. The white tick marks what ", term("Vegas", TIPS.vegas), " thinks."),
      el("div", { class: "week-bar" }, select, status), summary, cards);
    draw();
  }

  // ---------- Record ----------
  // Draw a label at the end of each chosen line, nudged apart vertically so close lines don't overlap.
  function drawEndLabels(chart, keep) {
    const ctx = chart.ctx; ctx.save(); ctx.font = "600 12px system-ui"; ctx.fillStyle = COLORS.text;
    const labels = [];
    chart.data.datasets.forEach((ds, i) => {
      if (!keep(ds)) return;
      const pts = chart.getDatasetMeta(i).data.filter((p) => !p.skip);
      const p = pts[pts.length - 1];
      if (p) labels.push({ text: ds.label, x: p.x + 6, y: p.y + 4 });
    });
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 14);
    labels.forEach((l) => ctx.fillText(l.text, l.x, l.y));
    ctx.restore();
  }
  const endLabels = (keep) => ({ id: "endLabels", afterDatasetsDraw: (chart) => drawEndLabels(chart, keep) });

  function renderRecord(root) {
    const r = D.record;
    root.append(el("h2", {}, "How we've done (live weeks only)"));
    if (!r.games) {
      root.append(el("p", { class: "big-line" }, "No live results yet."),
        el("p", { class: "sub" }, `Week ${D.meta.current_week} is locked and waiting for its games. The record fills in each week after Monday night.`));
    } else {
      const last = r.by_week[r.by_week.length - 1];
      root.append(
        el("p", { class: "big-line" }, `We picked ${last.ours} of ${last.games} winners (${pct(last.ours / last.games)}).`),
        el("p", { class: "sub" }, `Vegas picked ${last.vegas} of ${last.vegas_games} (${pct(last.vegas / Math.max(1, last.vegas_games))}). ` +
          "Over 2019–2025 our model averaged about 64%, Vegas about 66%."));
      if (r.by_week.length >= 1 && window.Chart) {
        const canvas = el("canvas", { "aria-label": "Share of winners picked correctly, by week, ours vs Vegas", role: "img" });
        root.append(el("div", { class: "chart-box" }, canvas));
        new Chart(canvas, {
          type: "line",
          data: { labels: r.by_week.map((w) => "Wk " + w.week), datasets: [
            { label: "Our model", data: r.by_week.map((w) => w.ours / w.games), borderColor: COLORS.ours, backgroundColor: COLORS.ours, borderWidth: 2, pointRadius: 4 },
            { label: "Vegas", data: r.by_week.map((w) => w.vegas / Math.max(1, w.vegas_games)), borderColor: COLORS.vegas, backgroundColor: COLORS.vegas, borderWidth: 2, pointRadius: 4 }] },
          options: { maintainAspectRatio: false, layout: { padding: { right: 70 } },
            interaction: { mode: "index", intersect: false },
            plugins: { legend: { labels: { color: COLORS.text } },
              tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.parsed.y)} right so far` } },
              title: { display: true, text: "Share of winners picked right, season so far", color: COLORS.text } },
            scales: { y: { min: 0.3, max: 0.9, ticks: { color: COLORS.text, callback: (v) => pct(v) }, grid: { color: COLORS.grid } },
                      x: { ticks: { color: COLORS.text }, grid: { display: false } } } },
          plugins: [endLabels(() => true)] });
      }
      if (r.extra.length) root.append(el("details", {}, el("summary", {}, "A bit more"),
        el("ul", {}, r.extra.map((line) => el("li", {}, line)))));
    }
    const bf = D.weeks.filter((w) => w.mode === "backfill").flatMap((w) => w.games)
      .filter((g) => g.result && g.result.winner !== null);
    if (bf.length) root.append(el("p", { class: "note" },
      `For reference only: in the backfill weeks (predicted after the fact), we picked ${bf.filter((g) => g.result.we_were_right).length} of ${bf.length} right. `,
      term("Why it doesn't count", TIPS.backfill)));
  }

  // ---------- Rankings ----------
  function renderRankings(root) {
    let rows = D.rankings.slice();
    const picked = rows.slice(0, 3).map((r) => r.team);  // default highlight: current top 3
    const cols = [
      { key: "rank", label: "#", num: true },
      { key: "team", label: "Team" },
      { key: "rating", label: "Rating", num: true, tip: TIPS.elo },
      { key: "rank_change", label: "Move", num: true },
      { key: "record", label: "Record" },
      { key: "next", label: "Next game" }];
    let sortKey = "rank", sortDir = 1;
    const tbody = el("tbody");
    const thead = el("thead", {}, el("tr", {}, cols.map((c) =>
      el("th", { class: c.num ? "num" : "", scope: "col", tabindex: "0",
        onclick: () => sortBy(c.key), onkeydown: (e) => { if (e.key === "Enter") sortBy(c.key); } },
        c.tip ? term(c.label, c.tip) : c.label))));
    function sortBy(key) {
      if (key === "next" || key === "record") return;
      sortDir = key === sortKey ? -sortDir : (key === "rating" || key === "rank_change" ? -1 : 1); sortKey = key;
      rows.sort((a, b) => (a[key] > b[key] ? 1 : a[key] < b[key] ? -1 : 0) * sortDir);
      thead.querySelectorAll("th").forEach((th, i) => th.setAttribute("aria-sort", cols[i].key === sortKey ? (sortDir > 0 ? "ascending" : "descending") : "none"));
      drawRows();
    }
    function moveCell(n) {
      return n > 0 ? el("span", { class: "up" }, "▲" + n) : n < 0 ? el("span", { class: "down" }, "▼" + -n) : el("span", { class: "flat" }, "—");
    }
    function drawRows() {
      tbody.replaceChildren(...rows.map((r) => el("tr", { class: picked.includes(r.team) ? "picked" : "", tabindex: "0",
        onclick: () => togglePick(r.team), onkeydown: (e) => { if (e.key === "Enter") togglePick(r.team); } },
        el("td", { class: "num" }, r.rank), el("td", {}, teamLabel(r.team)), el("td", { class: "num" }, Math.round(r.rating)),
        el("td", { class: "num" }, moveCell(r.rank_change)), el("td", {}, r.record),
        el("td", {}, r.next ? `${r.next.home ? "vs" : "@"} ${r.next.opp} · ${pct(r.next.p_win)}` : r.bye ? "Bye" : "—"))));
    }
    root.append(el("h2", {}, "Power rankings"),
      el("p", { class: "sub" }, "Every team's current ", term("Elo", TIPS.elo), " rating. \"Move\" is the change in rank since last week. Tap teams to compare them in the chart (up to 3)."),
      el("div", { class: "table-wrap" }, el("table", {}, thead, tbody)));
    drawRows();

    if (!window.Chart) return;
    const canvas = el("canvas", { role: "img", "aria-label": "Team ratings by week this season" });
    const legend = el("div", { class: "legend" });
    root.append(el("h3", {}, "Ratings through the season"), el("div", { class: "chart-box" }, canvas), legend);
    const weeks = [...new Set(Object.values(D.elo_history).flatMap((s) => s.map((p) => p[0])))].sort((a, b) => a - b);
    const chart = new Chart(canvas, {
      type: "line",
      data: { labels: weeks.map((w) => (w === 0 ? "Start" : "Wk " + w)), datasets: Object.entries(D.elo_history).map(([t, s]) => ({
        label: t, data: weeks.map((w) => { const p = s.find((x) => x[0] === w); return p ? p[1] : null; }), pointRadius: 0, tension: 0.2 })) },
      options: { maintainAspectRatio: false, layout: { padding: { right: 40 } }, animation: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: { legend: { display: false },
          tooltip: { filter: (item) => picked.includes(item.dataset.label), callbacks: { label: (c) => `${c.dataset.label}: ${Math.round(c.parsed.y)}` } } },
        scales: { y: { ticks: { color: COLORS.text }, grid: { color: COLORS.grid } }, x: { ticks: { color: COLORS.text }, grid: { display: false } } } },
      plugins: [endLabels((ds) => picked.includes(ds.label))] });
    function styleChart() {
      chart.data.datasets.forEach((ds) => {
        const i = picked.indexOf(ds.label);
        ds.borderColor = i >= 0 ? COLORS.pick[i] : "rgba(139,147,163,0.22)";
        ds.borderWidth = i >= 0 ? 2.5 : 1;
        ds.order = i >= 0 ? 0 : 1;
      });
      chart.update();
      legend.replaceChildren(...picked.map((t, i) => el("span", {}, el("i", { style: { background: COLORS.pick[i] } }), team(t).name)),
        el("span", {}, el("i", { style: { background: "rgba(139,147,163,0.5)" } }), "Other teams"));
    }
    function togglePick(t) {
      const i = picked.indexOf(t);
      if (i >= 0) picked.splice(i, 1); else { if (picked.length === 3) picked.shift(); picked.push(t); }
      drawRows(); styleChart();
    }
    styleChart();
  }

  // ---------- How It Works ----------
  function renderHow(root) {
    const card = (title, text) => el("div", { class: "how-card" }, el("h3", {}, title), el("p", {}, text));
    const bt = D.backtest;
    root.append(el("h2", {}, "How it works"),
      el("p", { class: "sub" }, "The short version, no math degree needed."),
      el("div", { class: "how-grid" },
        card("What is Elo?", "Every team has a rating. Win and your rating goes up; lose and it goes down. Beating a strong team (or losing to a weak one) moves it more. It only uses final scores."),
        card("What does 84% mean?", "If we say a team has an 84% chance, we expect it to win about 84 of every 100 games like that. It should still lose about 16 of them, so an upset doesn't mean the model is broken."),
        card("Why can't it beat Vegas?", "Vegas also knows about injuries, weather, and late news. Our model only knows scores. Over the past seven seasons it picked about 2 of every 3 winners, a little behind Vegas."),
        card("How do we keep it honest?", "Every week's picks are saved and committed to GitHub before the first kickoff, so they can't be changed after the fact. The 🔒 badge on each week links to that proof.")));

    const ex = D.example;
    if (ex) {
      const fav = ex.p_home >= 0.5 ? ex.home : ex.away;
      const won = ex.home_score > ex.away_score ? ex.home : ex.away;
      root.append(el("h3", {}, `Example: ${ex.away} at ${ex.home}, week ${ex.week}`), el("ol", { class: "steps" },
        el("li", {}, `Before the game: ${ex.home} was rated ${ex.home_rating}, ${ex.away} ${ex.away_rating}.`),
        el("li", {}, ex.home_field ? `${ex.home} gets ${ex.home_field} points for playing at home, so the gap is ${ex.gap} points.` : `Neutral site, so no home bonus: the gap is ${ex.gap} points.`),
        el("li", {}, `A ${Math.abs(ex.gap)}-point gap means about a ${pct(Math.max(ex.p_home, 1 - ex.p_home))} chance for ${fav}.`),
        el("li", {}, `Result: ${ex.home} ${ex.home_score}, ${ex.away} ${ex.away_score}. ${won} won, so ${ex.home} moved to ${ex.home_after} and ${ex.away} to ${ex.away_after}.`)));
    }
    if (bt && bt.games) root.append(el("p", { class: "note" },
      `Tested on ${bt.games.toLocaleString()} past games (2019–2025): our model picked ${pct(bt.elo)} of winners, Vegas ${pct(bt.vegas)}, and always picking the home team ${pct(bt.always_home)}.`));
    root.append(el("p", { class: "note" }, "This started as an attempt to beat the point spread. Public stats couldn't, so it became a win-chance model instead. ",
      el("a", { href: D.meta.repo + "#readme", target: "_blank", rel: "noopener" }, "The full story is on GitHub.")));

    root.append(el("h3", {}, "Glossary"), el("dl", { class: "glossary" },
      ...[["Elo rating", TIPS.elo], ["Chance to win", "How often we expect a team to win a game like this, out of 100."],
          ["Our line", TIPS.line], ["Vegas %", TIPS.vegas], ["Locked", TIPS.locked], ["Backfill", TIPS.backfill]]
        .flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])));
  }

  const RENDER = { week: renderWeek, record: renderRecord, rankings: renderRankings, how: renderHow };

  // ---------- boot ----------
  document.getElementById("repo-link").href = D.meta.repo;
  document.getElementById("generated").textContent = "Updated " + D.meta.generated;
  openTab(location.hash.slice(1));
})();
