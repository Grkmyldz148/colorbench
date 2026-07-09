const PROP_LABEL = {
  difference: "Difference", hue: "Hue", discrimination: "Discrimination",
  "3d_discrim": "3D discrim.", tolerance: "Tolerance", spacing: "Spacing",
};

fetch("leaderboard.json").then(r => r.json()).then(data => {
  document.getElementById("gen").textContent = data.generated;
  const props = data.properties;

  // property champion cards
  const cards = document.getElementById("winner-cards");
  props.forEach(p => {
    const champ = data.property_winners[p];
    const row = data.spaces.find(s => s.name === champ);
    const score = row && row.scores[p] != null ? row.scores[p].toFixed(2) : "—";
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `<div class="prop">${PROP_LABEL[p] || p}</div>
      <div class="champ">${champ || "—"}</div>
      <div class="score">${score}</div>`;
    cards.appendChild(el);
  });

  // per-property best/worst for coloring
  const stat = {};
  props.forEach(p => {
    const vals = data.spaces.map(s => s.scores[p]).filter(v => v != null);
    stat[p] = { min: Math.min(...vals), max: Math.max(...vals) };
  });

  const table = document.getElementById("leaderboard");
  const cols = [
    { key: "overall_rank", label: "Rank", get: s => s.overall_rank },
    { key: "name", label: "Color space", get: s => s.name },
    ...props.map(p => ({ key: p, label: PROP_LABEL[p] || p, get: s => s.scores[p] })),
  ];

  let sortKey = "overall_rank", asc = true;
  function render() {
    const rows = [...data.spaces].sort((a, b) => {
      const va = cols.find(c => c.key === sortKey).get(a);
      const vb = cols.find(c => c.key === sortKey).get(b);
      if (va == null) return 1; if (vb == null) return -1;
      if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      return asc ? va - vb : vb - va;
    });
    let html = "<thead><tr>" + cols.map(c =>
      `<th data-k="${c.key}" class="${c.key === sortKey ? "sorted" : ""}">${c.label}</th>`
    ).join("") + "</tr></thead><tbody>";
    rows.forEach((s, i) => {
      const helm = s.name.toLowerCase() === "helmlab";
      html += `<tr class="${helm ? "helm " : ""}${sortKey === "overall_rank" && i === 0 ? "rank-1" : ""}">`;
      cols.forEach(c => {
        let v = c.get(s), cls = "", disp = "";
        if (c.key === "name") { cls = "name"; disp = v; }
        else if (c.key === "overall_rank") { cls = "rank"; disp = v == null ? "—" : v; }
        else {
          disp = v == null ? "—" : v.toFixed(2);
          if (v != null) {
            if (Math.abs(v - stat[c.key].min) < 1e-9) cls = "best";
            else if (Math.abs(v - stat[c.key].max) < 1e-9) cls = "worst";
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
  render();
});
