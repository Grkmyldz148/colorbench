// Two boards (measurement / generation). Prefer embedded data so the page
// renders from file:// too (a bare fetch of a local JSON is CORS-blocked);
// fall back to fetch for freshness when served over http.
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

  function fmt(v, key) {
    if (v == null) return "—";
    if (key === "round_trip") return v.toExponential(1);
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(2);
  }

  function render() {
    const b = boards[current];
    titleEl.textContent = b.title;
    subEl.textContent = b.subtitle;
    if (b.holdout_note) { holdoutEl.hidden = false; holdoutEl.textContent = "⚠ " + b.holdout_note; }
    else holdoutEl.hidden = true;

    const metricCols = b.metrics;
    const cols = [
      { key: "overall_rank", label: "Rank", get: s => s.overall_rank },
      { key: "name", label: current === "measurement" ? "Model" : "Color space", get: s => s.name },
      ...metricCols.map(m => ({ key: m.key, label: m.label, get: s => s.scores[m.key] })),
    ];

    // per-column best/worst for coloring
    const stat = {};
    metricCols.forEach(m => {
      const vals = b.spaces.map(s => s.scores[m.key]).filter(v => v != null);
      stat[m.key] = { min: Math.min(...vals), max: Math.max(...vals) };
    });

    const rows = [...b.spaces].sort((a, z) => {
      const va = cols.find(c => c.key === sortKey).get(a);
      const vb = cols.find(c => c.key === sortKey).get(z);
      if (va == null) return 1; if (vb == null) return -1;
      if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      return asc ? va - vb : vb - va;
    });

    let html = "<thead><tr>" + cols.map(c =>
      `<th data-k="${c.key}" class="${c.key === sortKey ? "sorted" : ""}">${c.label}</th>`
    ).join("") + "</tr></thead><tbody>";
    rows.forEach((s, i) => {
      const helm = s.is_helm;
      html += `<tr class="${helm ? "helm " : ""}${sortKey === "overall_rank" && i === 0 ? "rank-1" : ""}">`;
      cols.forEach(c => {
        let v = c.get(s), cls = "", disp = "";
        if (c.key === "name") { cls = "name"; disp = v; }
        else if (c.key === "overall_rank") { cls = "rank"; disp = v == null ? "—" : v; }
        else {
          disp = fmt(v, c.key);
          if (v != null && stat[c.key]) {
            if (Math.abs(v - stat[c.key].min) < 1e-12) cls = "best";
            else if (Math.abs(v - stat[c.key].max) < 1e-12) cls = "worst";
          }
        }
        html += `<td class="${cls}">${disp}</td>`;
      });
      html += "</tr>";
    });
    table.innerHTML = html + "</tbody>";
    table.querySelectorAll("th").forEach(th => th.onclick = () => {
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
