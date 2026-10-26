/* Josh's NFL Elo: renders site/data/site-data.js (window.SITE_DATA). No framework.
   All text goes in via textContent, never innerHTML. */
(function () {
  "use strict";
  const D = window.SITE_DATA;
  // Light-surface chart colors (validated with the dataviz palette checker).
  const COLORS = { ours: "#2a78d6", vegas: "#eb6834", pick: ["#2a78d6", "#eb6834", "#1baf7a"],
                   grid: "#e9edf1", text: "#4a5361", other: "rgba(122,132,146,0.25)" };

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
  const team = (t) => D.teams[t] || { name: t, color: "#999" };
  const sq = (t) => el("span", { class: "sq", style: { background: team(t).color }, "aria-hidden": "true" });
  const term = (word, tip) => el("span", { class: "term", "data-tip": tip, tabindex: "0" }, word);

  const TIPS = {
    elo: "A rating for each team, built only from final scores. Win and it goes up; lose and it goes down. Beating a strong team moves it more.",
    line: "A point spread: how many points the favorite is expected to win by.",
    vegas: "The betting market's chance to win, worked out from the betting odds when we locked our picks.",
    locked: "Saved and committed to GitHub before the first game of the week kicked off, so it can't be changed after the fact.",
    backfill: "Weeks 1–5 were predicted after they were played (same rules, no peeking at results), so they don't count in the record.",
    gap: "We and Vegas differ by 15+ points. That usually means Vegas knows something our ratings don't, often a quarterback injury.",
    closing: "Vegas's final line right before kickoff.",
    rest: "Days since each team's last game. Shown when one team is much more rested (like coming off a bye). We tested adding it to our numbers; it didn't help, so it's info only.",
    proj: "Average final record when we play out the rest of the season 10,000 times with our win chances.",
    playoff: "How often the team made the playoffs in those 10,000 simulated seasons. Tap a number for division and #1-seed chances.",
    qbLock: "On Saturday, after teams post final injury reports, we re-lock the weekend games with QB news included. Also committed to GitHub before kickoff.",
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

  // ---------- sticky header height drives scroll-snap offsets ----------
  const header = document.getElementById("site-header");
  const setBarHeight = () => document.documentElement.style.setProperty("--bar-h", header.offsetHeight + "px");
  new ResizeObserver(setBarHeight).observe(header);

  // ---------- tabs (state lives in the URL hash so links are shareable) ----------
  const TABS = ["week", "record", "rankings", "how"];
  const rendered = {};
  function openTab(name) {
    if (!TABS.includes(name)) name = "week";
    for (const t of TABS) {
      document.getElementById("tab-" + t).hidden = t !== name;
      document.querySelector(`[data-tab="${t}"]`).setAttribute("aria-selected", String(t === name));
    }
    document.getElementById("week-bar").hidden = name !== "week";
    document.documentElement.classList.toggle("snap", name === "week");  // one game per screen only on This week
    if (!rendered[name]) { RENDER[name](document.getElementById("tab-" + name)); rendered[name] = true; }
    setBarHeight();
  }
  document.querySelectorAll("[data-tab]").forEach((b) =>
    b.addEventListener("click", () => { location.hash = b.dataset.tab; window.scrollTo(0, 0); }));
  window.addEventListener("hashchange", () => openTab(location.hash.slice(1)));

  // ---------- This week: one game per screen ----------
  // One side of the head-to-head: name, chance to win, and what we knew about the team at lock.
  function teamSide(s, cls, chance, isFav) {
    const metaBits = [s.record, `Elo ${s.elo}`].filter(Boolean).join(" · ");
    return el("section", { class: `side ${cls}` + (isFav ? " fav" : ""), "aria-label": team(s.team).name },
      el("div", { class: "team-head" },
        el("div", { class: "team-name" }, sq(s.team), team(s.team).name),
        el("div", { class: "chance", "aria-label": `${pct(chance)} chance to win` }, pct(chance))),
      el("div", { class: "meta" }, metaBits),
      el("div", { class: "meta last" }, s.last ? `Last: ${s.last}` : "First game of the season"),
      s.rest ? restGauge(s.team, s.rest, s.rested) : null);
  }

  // Rest gauge: one tick per day since the team's last game (max 14). Only on cards with a 3+ day gap; display only.
  function restGauge(t, days, rested) {
    const why = days >= 13 ? " · off a bye" : days <= 5 ? " · short week" : "";
    return el("div", { class: "rest" + (rested ? " more" : "") },
      el("span", { class: "rest-label", "data-tip": TIPS.rest, tabindex: "0" }, "Rest"),
      el("span", { class: "ticks", role: "img", "aria-label": `${days} days of rest${why}`,
                   style: rested ? { boxShadow: `0 0 0 1px ${team(t).color}55` } : null },
        Array.from({ length: 14 }, (_, i) => el("i", i < days ? { class: "on", style: { background: team(t).color } } : {}))),
      el("span", { class: "rest-days" }, `${days} days${why}`));
  }

  function gameCard(g) {
    const pHome = g.p_home, pAway = 1 - pHome;
    const favP = Math.max(pHome, pAway);
    const home = g.home, away = g.away;
    const rest = g.rest;
    const awaySide = { ...g.away_side, elo: g.why.away_rating, rest: rest && rest.away, rested: rest && rest.away > rest.home },
          homeSide = { ...g.home_side, elo: g.why.home_rating, rest: rest && rest.home, rested: rest && rest.home > rest.away };

    // Win-chance bar in team colors (underdog faded); the dark tick marks Vegas's view.
    const bar = el("div", { class: "bar", role: "img",
      "aria-label": `${away} ${pct(pAway)}, ${home} ${pct(pHome)}` + (g.p_vegas_home !== null ? `. Vegas: ${home} ${pct(g.p_vegas_home)}` : "") },
      el("div", { class: "seg" + (g.favorite === away ? "" : " dim"), style: { width: (pAway * 100) + "%", background: team(away).color } }),
      el("div", { class: "seg" + (g.favorite === home ? "" : " dim"), style: { width: (pHome * 100) + "%", background: team(home).color } }));
    const outer = el("div", { class: "bar-outer" }, bar);
    let legendRight = el("span", {}, "");
    if (g.p_vegas_home !== null) {
      outer.append(el("div", { class: "vegas-tick", style: { left: `calc(${(1 - g.p_vegas_home) * 100}% - 1.5px)` } }));
      const vFav = g.p_vegas_home >= 0.5 ? home : away;
      legendRight = el("span", {}, "▎ ", term("Vegas", TIPS.vegas), `: ${vFav} ${pct(Math.max(g.p_vegas_home, 1 - g.p_vegas_home))}`);
    }

    const lines = el("div", { class: "lines" },
      el("span", {}, term("Our line", TIPS.line), " ", el("b", {}, g.our_line || "—")),
      g.vegas_line_at_lock ? el("span", {}, "Vegas ", el("b", {}, g.vegas_line_at_lock), el("span", { class: "at" }, " at lock")) : null,
      g.closing_line ? el("span", {}, term("Closed", TIPS.closing), " ", el("b", {}, g.closing_line)) : null);

    // QB news. From Week 8 the pick accounts for it (in_pick); before that, or if the news came after the lock, it doesn't.
    const QB_REASON = { "injured reserve": "is out (injured reserve)", "reserve list": "is out (reserve list)",
                        out: "is ruled out", doubtful: "is listed as doubtful" };
    const qbAfter = (q) => !g.qb_adjusted ? "" : q.in_pick
      ? `Our pick accounts for this (about ${Math.round(g.why.qb_pts / 25)} points). `
      : "This came after our lock, so our pick doesn't account for it. ";
    const qbNote = (g.qb_out || []).length ? el("div", { class: "qb-note", role: "note" },
      el("b", {}, "Heads up: "),
      g.qb_out.map((q) => `${team(q.team).name}' usual starting QB${q.player ? ", " + q.player + "," : ""} ${QB_REASON[q.reason] || "is out"}. ${qbAfter(q)}`).join(""),
      g.qb_adjusted ? "" : "Our rating doesn't account for injuries; Vegas does.") : null;

    const w = g.why;
    const homeTotal = w.home_rating + w.home_field;
    const vegasFav = g.p_vegas_home === null ? null : (g.p_vegas_home >= 0.5 ? home : away);
    const vegasPct = g.p_vegas_home === null ? null : pct(Math.max(g.p_vegas_home, 1 - g.p_vegas_home));
    const why = el("details", { class: "why" }, el("summary", {}, "Why this pick"), el("ul", {},
      el("li", {}, w.home_field
        ? `${home} is rated ${w.home_rating}, plus ${w.home_field} for playing at home = ${homeTotal}. ${away} is rated ${w.away_rating}.`
        : `Neutral site, so no home-field bonus: ${home} ${w.home_rating} vs ${away} ${w.away_rating}.`),
      ...(w.qb_teams || []).map((t) => el("li", {},
        `${t}'s usual starting QB is out, so we take ${w.qb_pts} off ${t} (about ${Math.round(w.qb_pts / 25)} points on the scoreboard).`)),
      el("li", {}, `That's a ${Math.abs(w.edge)}-point edge for ${w.edge >= 0 ? home : away}, which works out to about ${pct(favP)} to win.`),
      vegasFav
        ? el("li", {}, g.big_gap
          ? `Vegas has ${vegasFav} at ${vegasPct}. A gap this big usually means news our ratings can't see, like ${g.qb_adjusted ? "other injuries" : "an injury"}.`
          : `Vegas sees it similarly: ${vegasFav} at ${vegasPct}.`)
        : null));

    // Result: we grade the winner only. The margin line shows how close our expected margin was (not graded).
    let result = null;
    if (g.result) {
      const r = g.result;
      const by = Math.abs(r.home_score - r.away_score);
      const exp = g.expected_margin === null ? null : Math.round(g.expected_margin);
      const expText = exp === null ? "" : exp < 1 ? " (we expected a toss-up)"
        : r.we_were_right ? ` (we expected ~${exp})` : ` (we expected ${g.favorite} by ~${exp})`;
      result = el("div", { class: "result" },
        el("span", {}, `Final: ${away} ${r.away_score}, ${home} ${r.home_score}`),
        r.winner === null ? el("span", {}, "Tie")
          : r.we_were_right ? el("span", { class: "right" }, "✓ Picked the winner") : el("span", { class: "wrong" }, "✗ Missed the winner"),
        r.winner === null ? null
          : el("span", { class: "margin" }, (r.we_were_right ? `Won by ${by}` : `${r.winner} won by ${by}`) + expText));
    }

    return el("article", { class: "game", "aria-label": `${team(away).name} at ${team(home).name}` },
      el("div", { class: "game-card" },
        el("div", { class: "game-top" },
          el("div", {}, el("span", { class: "slot" }, g.day_group), " · ", g.kickoff, g.neutral ? " · Neutral site" : ""),
          el("div", { class: "tags" },
            g.big_gap ? el("span", { class: "pill gap", "data-tip": TIPS.gap, tabindex: "0" }, "Big gap with Vegas") : null,
            el("span", { class: "pill" + (g.confidence === "Toss-up" ? " toss" : "") }, g.confidence))),
        el("div", { class: "h2h" },
          teamSide(awaySide, "away", pAway, g.favorite === away),
          teamSide(homeSide, "home", pHome, g.favorite === home)),
        el("div", { class: "bar-wrap" }, outer,
          el("div", { class: "bar-legend" }, el("span", {}, `${team(g.favorite).name} favored`), legendRight)),
        lines, qbNote, why, result));
  }

  function renderWeek(root) {
    const weeks = D.weeks;
    const bar = document.getElementById("week-bar");
    if (!weeks.length) { root.append(el("p", {}, "No predictions yet.")); return; }
    const select = el("select", { id: "week-select" },
      weeks.map((w) => el("option", { value: w.week, selected: w.week === D.meta.current_week },
        `Week ${w.week}` + (w.mode === "backfill" ? " · backfill" : ""))));
    const status = el("span", {});
    bar.append(el("label", { for: "week-select" }, "Week"), select, status);
    const intro = el("div", { class: "week-intro" });
    const list = el("div", {});
    root.append(intro, list);

    function draw() {
      const w = weeks.find((x) => x.week === Number(select.value));
      status.replaceChildren(
        w.mode === "live" && w.lock
          ? (w.lock.locked
            ? el("a", { class: "lock" + (w.lock.before_kickoff ? "" : " late"), href: w.lock.url, target: "_blank", rel: "noopener",
                "data-tip": TIPS.locked }, w.lock.text)
            : el("span", { class: "backfill-note" }, "Not locked yet"))
          : el("span", { class: "backfill-note" }, "Backfill: picked after the games, doesn't count. ",
              term("Why?", TIPS.backfill)),
        w.qb_lock && w.qb_lock.locked
          ? el("a", { class: "lock" + (w.qb_lock.before_kickoff ? "" : " late"), href: w.qb_lock.url, target: "_blank",
              rel: "noopener", "data-tip": TIPS.qbLock }, w.qb_lock.text)
          : "");
      const done = w.games.filter((g) => g.result && g.result.winner !== null);
      let summary = `${w.games.length} games. Scroll for one game at a time.`;
      if (done.length) {
        const ours = done.filter((g) => g.result.we_were_right).length;
        const vDone = done.filter((g) => g.result.vegas_was_right !== null);
        summary = `We picked ${ours} of ${done.length} winners this week. Vegas: ${vDone.filter((g) => g.result.vegas_was_right).length} of ${vDone.length}.`;
      }
      intro.replaceChildren(
        el("h2", {}, `Week ${w.week}`),
        el("p", {}, "Each game shows who our ", term("Elo", TIPS.elo),
          " ratings favor and their chance to win, next to what Vegas thinks."),
        el("p", { class: "hint" }, summary));
      list.replaceChildren(...w.games.map(gameCard));
      setBarHeight();
    }
    select.addEventListener("change", () => { draw(); window.scrollTo(0, 0); });
    draw();
  }

  // ---------- Record ----------
  // Label each chosen line at its last point, nudged apart so close lines don't overlap.
  function drawEndLabels(chart, keep) {
    const ctx = chart.ctx; ctx.save(); ctx.font = "500 12px Geist, system-ui"; ctx.fillStyle = COLORS.text;
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
  const axis = (extra) => ({ ticks: { color: COLORS.text, font: { family: "Geist" } }, grid: { color: COLORS.grid }, border: { display: false }, ...extra });

  function renderRecord(root) {
    const r = D.record;
    root.append(el("div", { class: "panel-head" }, el("h2", {}, "Record"),
      el("p", {}, "How our locked picks have done, live weeks only.")));
    if (!r.games) {
      root.append(el("p", { class: "big-line" }, "No live results yet."),
        el("p", { class: "sub" }, `Week ${D.meta.current_week} is locked and waiting for its games. The record fills in each week after Monday night.`));
    } else {
      const last = r.by_week[r.by_week.length - 1];
      root.append(
        el("p", { class: "big-line" }, `We picked ${last.ours} of ${last.games} winners (${pct(last.ours / last.games)}).`),
        el("p", { class: "sub" }, `Vegas picked ${last.vegas} of ${last.vegas_games} (${pct(last.vegas / Math.max(1, last.vegas_games))}). ` +
          "Over 2019–2025 our model averaged about 64%, Vegas about 66%."));
      if (window.Chart) {
        const canvas = el("canvas", { "aria-label": "Share of winners picked correctly, by week, ours vs Vegas", role: "img" });
        root.append(el("div", { class: "surface chart-box" }, canvas));
        new Chart(canvas, {
          type: "line",
          data: { labels: r.by_week.map((w) => "Wk " + w.week), datasets: [
            { label: "Our model", data: r.by_week.map((w) => w.ours / w.games), borderColor: COLORS.ours, backgroundColor: COLORS.ours, borderWidth: 2, pointRadius: 4 },
            { label: "Vegas", data: r.by_week.map((w) => w.vegas / Math.max(1, w.vegas_games)), borderColor: COLORS.vegas, backgroundColor: COLORS.vegas, borderWidth: 2, pointRadius: 4 }] },
          options: { maintainAspectRatio: false, layout: { padding: { right: 72 } },
            interaction: { mode: "index", intersect: false },
            plugins: { legend: { labels: { color: COLORS.text, boxWidth: 14, boxHeight: 3 } },
              tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.parsed.y)} right so far` } } },
            scales: { y: axis({ min: 0.3, max: 0.9, ticks: { color: COLORS.text, callback: (v) => pct(v) } }), x: axis({ grid: { display: false } }) } },
          plugins: [endLabels(() => true)] });
      }
      if (r.extra.length) root.append(el("details", { class: "surface more" }, el("summary", {}, "A bit more"),
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
    let rows = D.rankings.map((r) => ({ ...r, proj: r.odds ? r.odds.wins : null, playoff: r.odds ? r.odds.playoff : null }));
    const picked = rows.slice(0, 3).map((r) => r.team);
    const cols = [
      { key: "rank", label: "#", num: true }, { key: "team", label: "Team" },
      { key: "rating", label: "Rating", num: true, tip: TIPS.elo }, { key: "rank_change", label: "Move", num: true, cls: "move" },
      { key: "record", label: "Record" }, { key: "proj", label: "Proj. record", num: true, tip: TIPS.proj, cls: "proj" },
      { key: "playoff", label: "Playoffs", num: true, tip: TIPS.playoff }, { key: "next", label: "Next game", cls: "next" }];
    let sortKey = "rank", sortDir = 1;
    const tbody = el("tbody");
    const thead = el("thead", {}, el("tr", {}, cols.map((c) =>
      el("th", { class: (c.num ? "num " : "") + (c.cls || ""), scope: "col", tabindex: "0",
        onclick: () => sortBy(c.key), onkeydown: (e) => { if (e.key === "Enter") sortBy(c.key); } },
        c.tip ? term(c.label, c.tip) : c.label))));
    function sortBy(key) {
      if (key === "next" || key === "record") return;
      sortDir = key === sortKey ? -sortDir : (["rating", "rank_change", "proj", "playoff"].includes(key) ? -1 : 1); sortKey = key;
      rows.sort((a, b) => (a[key] > b[key] ? 1 : a[key] < b[key] ? -1 : 0) * sortDir);
      thead.querySelectorAll("th").forEach((th, i) => th.setAttribute("aria-sort", cols[i].key === sortKey ? (sortDir > 0 ? "ascending" : "descending") : "none"));
      drawRows();
    }
    const oddsPct = (x) => x < 0.005 ? "<1%" : x > 0.995 ? ">99%" : pct(x);
    const projCell = (o) => { if (!o) return "—"; const games = Math.round(o.wins + o.losses), w = Math.round(o.wins); return `${w}-${games - w}`; };
    const playoffCell = (o) => !o ? "—" : el("span", { class: "term", tabindex: "0",
      "data-tip": `Win the division: ${oddsPct(o.division)} · #1 seed (first-round bye): ${oddsPct(o.top_seed)}` }, oddsPct(o.playoff));
    const moveCell = (n) => n > 0 ? el("span", { class: "up" }, "▲" + n) : n < 0 ? el("span", { class: "down" }, "▼" + -n) : el("span", { class: "flat" }, "—");
    function drawRows() {
      tbody.replaceChildren(...rows.map((r) => el("tr", { class: picked.includes(r.team) ? "picked" : "", tabindex: "0",
        onclick: () => togglePick(r.team), onkeydown: (e) => { if (e.key === "Enter") togglePick(r.team); } },
        el("td", { class: "num" }, r.rank),
        el("td", {}, el("span", { class: "team-cell", title: team(r.team).name }, sq(r.team),
          el("span", { class: "full" }, team(r.team).name), el("span", { class: "abbr" }, r.team))),
        el("td", { class: "num" }, Math.round(r.rating)), el("td", { class: "num move" }, moveCell(r.rank_change)),
        el("td", {}, r.record),
        el("td", { class: "num proj" }, projCell(r.odds)), el("td", { class: "num" }, playoffCell(r.odds)),
        el("td", { class: "next" }, r.next ? `${r.next.home ? "vs" : "@"} ${r.next.opp} · ${pct(r.next.p_win)}` : r.bye ? "Bye" : "—"))));
    }
    root.append(el("div", { class: "panel-head" }, el("h2", {}, "Power rankings"),
      el("p", {}, "Every team's current ", term("Elo", TIPS.elo), " rating. \"Move\" is the change in rank since last week. Tap teams to compare them in the chart (up to 3).")),
      el("div", { class: "surface table-wrap" }, el("table", {}, thead, tbody)));
    drawRows();

    if (!window.Chart) return;
    const canvas = el("canvas", { role: "img", "aria-label": "Team ratings by week this season" });
    const legend = el("div", { class: "legend" });
    root.append(el("h3", { class: "section" }, "Ratings through the season"), el("div", { class: "surface chart-box" }, canvas), legend);
    const weeks = [...new Set(Object.values(D.elo_history).flatMap((s) => s.map((p) => p[0])))].sort((a, b) => a - b);
    const chart = new Chart(canvas, {
      type: "line",
      data: { labels: weeks.map((w) => (w === 0 ? "Start" : "Wk " + w)), datasets: Object.entries(D.elo_history).map(([t, s]) => ({
        label: t, data: weeks.map((w) => { const p = s.find((x) => x[0] === w); return p ? p[1] : null; }), pointRadius: 0, tension: 0.2 })) },
      options: { maintainAspectRatio: false, layout: { padding: { right: 44 } }, animation: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: { legend: { display: false },
          tooltip: { filter: (item) => picked.includes(item.dataset.label), callbacks: { label: (c) => `${c.dataset.label}: ${Math.round(c.parsed.y)}` } } },
        scales: { y: axis(), x: axis({ grid: { display: false } }) } },
      plugins: [endLabels((ds) => picked.includes(ds.label))] });
    function styleChart() {
      chart.data.datasets.forEach((ds) => {
        const i = picked.indexOf(ds.label);
        ds.borderColor = i >= 0 ? COLORS.pick[i] : COLORS.other;
        ds.borderWidth = i >= 0 ? 2.5 : 1;
        ds.order = i >= 0 ? 0 : 1;
      });
      chart.update();
      legend.replaceChildren(...picked.map((t, i) => el("span", {}, el("i", { style: { background: COLORS.pick[i] } }), team(t).name)),
        el("span", {}, el("i", { style: { background: "rgba(122,132,146,0.5)" } }), "Other teams"));
    }
    function togglePick(t) {
      const i = picked.indexOf(t);
      if (i >= 0) picked.splice(i, 1); else { if (picked.length === 3) picked.shift(); picked.push(t); }
      drawRows(); styleChart();
    }
    styleChart();
  }

  // ---------- How it works ----------
  function renderHow(root) {
    const card = (title, text) => el("div", { class: "surface how-card" }, el("h3", {}, title), el("p", {}, text));
    const bt = D.backtest;
    root.append(el("div", { class: "panel-head" }, el("h2", {}, "How it works"), el("p", {}, "The short version, no math degree needed.")),
      el("div", { class: "how-grid" },
        card("What is Elo?", "Every team has a rating. Win and your rating goes up; lose and it goes down. Beating a strong team (or losing to a weak one) moves it more. It only uses final scores."),
        card("What does 84% mean?", "If we say a team has an 84% chance, we expect it to win about 84 of every 100 games like that. It should still lose about 16 of them, so an upset doesn't mean the model is broken."),
        card("What about injuries?", "Since Week 8, if a team's usual starting quarterback is out (ruled out, doubtful, or on injured reserve), we take about 3 to 4 points off that team. In testing on 2019–2025 this helped a little, and it picked more winners in games with a QB out. Other injuries aren't counted."),
        card("Why can't it beat Vegas?", "Vegas also knows about other injuries, weather, and late news. Our model knows scores and starting quarterbacks. Over the past seven seasons it picked about 2 of every 3 winners, a little behind Vegas."),
        card("Playoff odds", "We play out the rest of the season 10,000 times using our win chances, updating ratings as the simulated games go. Tiebreakers are simplified (ties in the standings are broken at random). Checked on 2019–2025: teams we gave 75%+ at midseason made it about 84% of the time."),
        card("How do we keep it honest?", "Every week's picks are saved and committed to GitHub before the first kickoff, so they can't be changed after the fact. The “Locked” note next to each week links to that proof.")));
    const ex = D.example;
    if (ex) {
      const fav = ex.p_home >= 0.5 ? ex.home : ex.away;
      const won = ex.home_score > ex.away_score ? ex.home : ex.away;
      root.append(el("h3", { class: "section" }, `Example: ${team(ex.away).name} at ${team(ex.home).name}, week ${ex.week}`), el("ol", { class: "surface steps" },
        el("li", {}, `Before the game: ${ex.home} was rated ${ex.home_rating}, ${ex.away} ${ex.away_rating}.`),
        el("li", {}, ex.home_field ? `${ex.home} gets ${ex.home_field} points for playing at home, so the gap is ${ex.gap} points.` : `Neutral site, so no home bonus: the gap is ${ex.gap} points.`),
        el("li", {}, `A ${Math.abs(ex.gap)}-point gap means about a ${pct(Math.max(ex.p_home, 1 - ex.p_home))} chance for ${fav}.`),
        el("li", {}, `Result: ${ex.home} ${ex.home_score}, ${ex.away} ${ex.away_score}. ${won} won, so ${ex.home} moved to ${ex.home_after} and ${ex.away} to ${ex.away_after}.`)));
    }
    if (bt && bt.games) root.append(el("p", { class: "note" },
      `Tested on ${bt.games.toLocaleString()} past games (2019–2025): our model picked ${pct(bt.elo)} of winners, Vegas ${pct(bt.vegas)}, and always picking the home team ${pct(bt.always_home)}.`));
    root.append(el("p", { class: "note" }, "This started as an attempt to beat the point spread. Public stats couldn't, so it became a win-chance model instead. ",
      el("a", { href: D.meta.repo + "#readme", target: "_blank", rel: "noopener" }, "The full story is on GitHub.")));
    root.append(el("h3", { class: "section" }, "Glossary"), el("dl", { class: "surface glossary" },
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
