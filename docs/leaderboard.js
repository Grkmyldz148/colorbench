// Two boards (measurement / generation), each a detailed grouped comparison.
// Prefer embedded data so the page renders from file:// too (a bare fetch of a
// local JSON is CORS-blocked); fall back to fetch when served over http.
function loadData() {
  if (window.LEADERBOARD) return Promise.resolve(window.LEADERBOARD);
  return fetch("leaderboard.json").then(r => r.json());
}

loadData().then(data => {
  document.getElementById("gen").textContent = data.generated;
  const boards = data.boards;
  let current = "measurement";
  let sortKey = "overall_rank", asc = true;

  const titleEl = document.getElementById("board-title");
  const subEl = document.getElementById("board-sub");
  const holdoutEl = document.getElementById("board-holdout");
  const table = document.getElementById("leaderboard");

  function fmt(v, key, signed) {
    if (v == null) return "";
    if (String(key).startsWith("rt_")) return v.toExponential(1);
    if (signed) return (v > 0 ? "+" : "") + v;
    if (Number.isInteger(v)) return String(v);
    return Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(2);
  }

  function render() {
    const b = boards[current];
    titleEl.textContent = b.title;
    subEl.textContent = b.subtitle;
    if (b.holdout_note) { holdoutEl.hidden = false; holdoutEl.textContent = "⚠ " + b.holdout_note; }
    else holdoutEl.hidden = true;

    // flatten metrics from groups, keep group spans for the header
    const groups = b.groups;
    const metricCols = [];
    groups.forEach(g => g.metrics.forEach(m =>
      metricCols.push({ ...m, group: g.label, scored: g.scored !== false })));

    // best/worst tint only on SCORED, unsigned columns - diagnostic (CIELab-
    // referenced) columns are shown plain so they don't imply a winner
    const stat = {};
    metricCols.forEach(m => {
      if (m.signed || !m.scored) return;
      const vals = b.spaces.map(s => s.scores[m.key]).filter(v => v != null);
      if (vals.length) stat[m.key] = { min: Math.min(...vals), max: Math.max(...vals) };
    });

    const getVal = (s, key) => key === "overall_rank" ? s.overall_rank
      : key === "name" ? s.name : s.scores[key];

    const rows = [...b.spaces].sort((a, z) => {
      const va = getVal(a, sortKey), vb = getVal(z, sortKey);
      if (va == null) return 1; if (vb == null) return -1;
      if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      return asc ? va - vb : vb - va;
    });

    // ── header row 1: group spans ; row 2: metric labels ──
    // cfreeze1/cfreeze2 mark the ONLY frozen cells (rank + name). Using classes
    // rather than :first-child avoids freezing row-2's first metric cell by
    // accident (the bug that misaligned the grouped header on scroll).
    let h1 = `<tr><th class="grp cfreeze1" rowspan="2" data-k="overall_rank">Rank</th>`
           + `<th class="grp cfreeze2" rowspan="2" data-k="name">${current === "measurement" ? "Model" : "Color space"}</th>`;
    let h2 = "<tr>";
    groups.forEach(g => {
      const dg = g.scored === false ? " diag" : "";
      h1 += `<th class="grolabel${dg}" colspan="${g.metrics.length}">${g.label}</th>`;
      g.metrics.forEach(m => {
        const t = m.hint ? ` title="${m.hint}"` : "";
        h2 += `<th data-k="${m.key}" class="${m.key === sortKey ? "sorted" : ""}${dg}"${t}>${m.label}${m.hint ? " ⓘ" : ""}</th>`;
      });
    });
    h1 += "</tr>"; h2 += "</tr>";

    let body = "<tbody>";
    rows.forEach((s, i) => {
      body += `<tr class="${sortKey === "overall_rank" && i === 0 ? "rank-1" : ""}">`;
      body += `<td class="rank cfreeze1">${s.overall_rank == null ? "" : s.overall_rank}</td>`;
      body += `<td class="name cfreeze2">${s.name}</td>`;
      metricCols.forEach(m => {
        const v = s.scores[m.key];
        let cls = m.scored ? "" : "diag";
        if (m.scored && !m.signed && v != null && stat[m.key]) {
          if (Math.abs(v - stat[m.key].min) < 1e-12) cls = "best";
          else if (Math.abs(v - stat[m.key].max) < 1e-12) cls = "worst";
        }
        const ci = s.ci && s.ci[m.key];
        const tip = ci ? ` title="CI95 ${ci[0]}-${ci[1]}"` : "";
        body += `<td class="${cls}"${tip}>${fmt(v, m.key, m.signed)}</td>`;
      });
      body += "</tr>";
    });
    body += "</tbody>";

    table.innerHTML = "<thead>" + h1 + h2 + "</thead>" + body;
    table.querySelectorAll("th[data-k]").forEach(th => th.onclick = () => {
      const k = th.dataset.k;
      if (k === sortKey) asc = !asc; else { sortKey = k; asc = true; }
      render();
    });
  }

  document.querySelectorAll("#tabs button").forEach(btn => btn.onclick = () => {
    document.querySelectorAll("#tabs button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    current = btn.dataset.board;
    sortKey = "overall_rank"; asc = true;
    render();
  });

  render();
});
