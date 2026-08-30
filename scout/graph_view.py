"""The Graph page's canvas — a live force-directed map of the knowledge graph.

Pure builders: edge rows (store.all_graph_edges) in, a self-contained HTML
document out, rendered by the UI in a components iframe. No external JS —
the layout is a small custom simulation, so the page works offline and adds
no dependency.

Colors are Scout's warm-paper system extended with one validated categorical
set (5 node types, checked for CVD separation and contrast against the
surface — see the palette block). Shape is the secondary encoding: identity
never rides on color alone. Node labels and evidence strings are UNTRUSTED
text (extracted from bios, websites, research) — they enter the page only
via JSON with `</` escaped, and are rendered with canvas fillText /
textContent, never as HTML.
"""

from __future__ import annotations

import json

# Scout's chart surface + inks (matches ui.py's :root).
SURFACE = "#fbf5ec"
INK = "#20180f"
MUTED = "#6b6052"
HAIR = "rgba(32,24,15,0.14)"

# Categorical node palette — validated (lightness band, chroma floor, CVD
# adjacent-pair separation, normal-vision floor, 3:1 contrast on SURFACE).
# Fixed assignment by TYPE, never by count or rank; shapes are the secondary
# encoding for the CVD/print case.
NODE_STYLE = {
    "company": {"color": "#a37b10", "shape": "circle", "label": "Company"},
    "investor": {"color": "#4a63d8", "shape": "diamond", "label": "Investor"},
    "person": {"color": "#b03a68", "shape": "dot", "label": "Person"},
    "lab": {"color": "#5c7f1f", "shape": "square", "label": "Lab"},
    "watcher": {"color": "#8a4bc9", "shape": "ring", "label": "Watcher"},
}
TYPE_ORDER = ["company", "investor", "person", "lab", "watcher"]

# Edge dash per relationship — recessive by default, colored on selection.
REL_DASH = {
    "invested_in": [],
    "founded": [],
    "acquired_by": [],
    "alum_of": [6, 4],
    "follows": [2, 4],
}


def graph_data(
    edges: list[dict],
    rels: set[str] | None = None,
    cross_links_only: bool = False,
    focus_key: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """(nodes, links) for the canvas, filtered. Pure — unit-tested.

    `cross_links_only` drops connector nodes (everything but companies) that
    touch a single company: a one-edge investor restates the card's own
    chips, and a few hundred of them bury the pattern the graph exists to
    show. `focus_key` keeps the 2-hop neighborhood of one node.
    """
    rows = [e for e in edges if rels is None or e["rel"] in rels]

    if focus_key:
        keep = {focus_key}
        # hop 1: everything touching the focus; hop 2: everything touching that.
        for _ in range(2):
            grown = set(keep)
            for e in rows:
                if e["src_key"] in keep or e["dst_key"] in keep:
                    grown.add(e["src_key"])
                    grown.add(e["dst_key"])
            keep = grown
        rows = [e for e in rows if e["src_key"] in keep and e["dst_key"] in keep]

    if cross_links_only:
        degree: dict[str, int] = {}
        types: dict[str, str] = {}
        for e in rows:
            degree[e["src_key"]] = degree.get(e["src_key"], 0) + 1
            degree[e["dst_key"]] = degree.get(e["dst_key"], 0) + 1
            types[e["src_key"]] = e["src_type"]
            types[e["dst_key"]] = e["dst_type"]
        connectors_kept = {
            k for k, t in types.items()
            if t == "company" or degree.get(k, 0) >= 2 or k == focus_key
        }
        rows = [e for e in rows
                if e["src_key"] in connectors_kept and e["dst_key"] in connectors_kept]

    index: dict[str, int] = {}
    nodes: list[dict] = []

    def node_of(key: str, label: str, ntype: str) -> int:
        if key not in index:
            index[key] = len(nodes)
            nodes.append({"key": key, "label": label,
                          "type": ntype if ntype in NODE_STYLE else "person",
                          "degree": 0})
        return index[key]

    links: list[dict] = []
    for e in rows:
        s = node_of(e["src_key"], e["src_label"], e["src_type"])
        d = node_of(e["dst_key"], e["dst_label"], e["dst_type"])
        if s == d:
            continue
        nodes[s]["degree"] += 1
        nodes[d]["degree"] += 1
        links.append({"s": s, "d": d, "rel": e["rel"],
                      "evidence": (e.get("evidence") or "")[:160]})
    return nodes, links


def _js_payload(value) -> str:
    """JSON safe to inline inside a <script> block: `</` escaped so an
    adversarial label ("</script><script>…") extracted from a bio or a
    website can never break out of the data literal."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def graph_page_html(nodes: list[dict], links: list[dict],
                    height: int = 620, focus_key: str | None = None) -> str:
    """The self-contained document for st.components.v1.html."""
    styles = {t: {"color": v["color"], "shape": v["shape"]}
              for t, v in NODE_STYLE.items()}
    legend = "".join(
        f'<span class="lg"><canvas class="sw" data-type="{t}" width="14" '
        f'height="14"></canvas>{NODE_STYLE[t]["label"]}</span>'
        for t in TYPE_ORDER
    )
    return f"""
<div id="wrap">
  <div id="bar">{legend}
    <span id="hint">drag nodes · scroll to zoom · click to trace · double-click to release</span>
  </div>
  <canvas id="net"></canvas>
  <div id="info" hidden></div>
</div>
<style>
  * {{ box-sizing: border-box; margin: 0; }}
  #wrap {{ position: relative; width: 100%; height: {height}px;
          font: 13px -apple-system, "Segoe UI", sans-serif;
          background: {SURFACE}; border: 1px solid {HAIR}; border-radius: 10px;
          overflow: hidden; }}
  #net {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
  #bar {{ position: absolute; top: 0; left: 0; right: 0; z-index: 2;
         display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
         padding: 8px 12px; color: {MUTED};
         background: linear-gradient({SURFACE}ee, {SURFACE}00);
         pointer-events: none; }}
  .lg {{ display: inline-flex; gap: 5px; align-items: center; color: {INK}; }}
  .sw {{ width: 14px; height: 14px; }}
  #hint {{ margin-left: auto; font-size: 11px; }}
  #info {{ position: absolute; left: 10px; bottom: 10px; z-index: 2;
          max-width: 46%; max-height: 45%; overflow: auto; padding: 10px 12px;
          background: {SURFACE}; border: 1px solid {HAIR}; border-radius: 8px;
          color: {INK}; box-shadow: 0 2px 10px rgba(32,24,15,0.10); }}
  #info h4 {{ font-size: 13px; margin-bottom: 4px; }}
  #info .t {{ color: {MUTED}; font-size: 11px; }}
  #info .e {{ margin-top: 4px; font-size: 12px; }}
  #info .ev {{ color: {MUTED}; font-size: 11px; }}
</style>
<script>
const NODES = {_js_payload(nodes)};
const LINKS = {_js_payload(links)};
const STYLE = {_js_payload(styles)};
const RELDASH = {_js_payload(REL_DASH)};
const RELLABEL = {{invested_in: "backs", founded: "founded", alum_of: "alum of",
                 acquired_by: "acquired by", follows: "follows"}};
const FOCUS = {_js_payload(focus_key)};
const SURFACE = "{SURFACE}", INK = "{INK}", MUTED = "{MUTED}";

// Legend swatches carry the shape too — identity never rides on color alone.
function glyph(g, x, y, r, type, fill) {{
  const c = STYLE[type].color;
  g.strokeStyle = c; g.fillStyle = fill ? c : SURFACE; g.lineWidth = 1.6;
  g.beginPath();
  const shape = STYLE[type].shape;
  if (shape === "diamond") {{
    g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y);
    g.closePath();
  }} else if (shape === "square") {{
    g.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7);
  }} else {{
    g.arc(x, y, shape === "dot" ? Math.max(r * 0.8, 2.5) : r, 0, 7);
  }}
  if (STYLE[type].shape === "ring") {{ g.fillStyle = SURFACE; }}
  g.fill(); g.stroke();
}}
document.querySelectorAll(".sw").forEach(sw => {{
  const g = sw.getContext("2d");
  glyph(g, 7, 7, 5, sw.dataset.type, true);
}});

const canvas = document.getElementById("net");
const info = document.getElementById("info");
const ctx = canvas.getContext("2d");
let W = 0, H = 0, DPR = window.devicePixelRatio || 1;
function resize() {{
  W = canvas.clientWidth; H = canvas.clientHeight;
  canvas.width = W * DPR; canvas.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}}
resize(); window.addEventListener("resize", () => {{ resize(); draw(); }});

// ---- layout state ----------------------------------------------------------
const N = NODES.length;
for (let i = 0; i < N; i++) {{
  const a = 2 * Math.PI * i / Math.max(N, 1);
  const rr = Math.min(W, H) * 0.42 * (0.5 + 0.5 * Math.random());
  NODES[i].x = W / 2 + rr * Math.cos(a);
  NODES[i].y = H / 2 + rr * Math.sin(a);
  NODES[i].vx = 0; NODES[i].vy = 0;
  NODES[i].r = (NODES[i].type === "company" ? 7 : 5)
             + 2.2 * Math.sqrt(Math.max(NODES[i].degree - 1, 0));
}}
const adj = NODES.map(() => []);
LINKS.forEach((l, i) => {{ adj[l.s].push(i); adj[l.d].push(i); }});

// The label layer: always the focus + the most connected; hover/selection
// adds the rest on demand. A label on every node is soup, not a map.
const byDegree = [...NODES.keys()].sort((a, b) => NODES[b].degree - NODES[a].degree);
const labelled = new Set(byDegree.slice(0, Math.min(18, N)));
if (FOCUS) NODES.forEach((n, i) => {{ if (n.key === FOCUS) labelled.add(i); }});

// ---- physics: springs + sampled repulsion + gravity ------------------------
let alpha = 1.0;
function tick() {{
  const K = 0.02, REST = 128, GRAV = 0.005;
  for (const l of LINKS) {{
    const a = NODES[l.s], b = NODES[l.d];
    const dx = b.x - a.x, dy = b.y - a.y;
    const dist = Math.hypot(dx, dy) || 1;
    const f = K * (dist - REST) / dist;
    a.vx += f * dx; a.vy += f * dy; b.vx -= f * dx; b.vy -= f * dy;
  }}
  // O(n·s) sampled repulsion keeps big graphs fluid; small graphs get exact.
  const S = N > 220 ? 24 : N;
  for (let i = 0; i < N; i++) {{
    const a = NODES[i];
    for (let k = 0; k < S; k++) {{
      const j = (S === N) ? k : (Math.random() * N) | 0;
      if (i === j) continue;
      const b = NODES[j];
      const dx = a.x - b.x, dy = a.y - b.y;
      const d2 = dx * dx + dy * dy + 40;
      const f = 2800 / d2 * (S === N ? 1 : N / S * 0.5);
      a.vx += f * dx / Math.sqrt(d2); a.vy += f * dy / Math.sqrt(d2);
    }}
    a.vx += (W / 2 - a.x) * GRAV; a.vy += (H / 2 - a.y) * GRAV;
  }}
  for (const n of NODES) {{
    if (n === dragging) continue;
    n.x += (n.vx *= 0.62) * alpha; n.y += (n.vy *= 0.62) * alpha;
  }}
  alpha = Math.max(alpha * 0.996, 0.03);
}}

// ---- viewport --------------------------------------------------------------
let zoom = 1, panX = 0, panY = 0;
const toScreen = n => [n.x * zoom + panX, n.y * zoom + panY];
const toWorld = (x, y) => [(x - panX) / zoom, (y - panY) / zoom];

let hovered = -1, selected = -1, dragging = null, panning = null;

function neighbourhood(i) {{
  const nodes = new Set([i]), links = new Set();
  for (const li of adj[i]) {{
    links.add(li);
    nodes.add(LINKS[li].s); nodes.add(LINKS[li].d);
  }}
  return {{ nodes, links }};
}}

function draw() {{
  ctx.clearRect(0, 0, W, H);
  const sel = selected >= 0 ? neighbourhood(selected) : null;
  ctx.lineCap = "round";
  for (let li = 0; li < LINKS.length; li++) {{
    const l = LINKS[li];
    const inSel = sel && sel.links.has(li);
    const [x1, y1] = toScreen(NODES[l.s]), [x2, y2] = toScreen(NODES[l.d]);
    ctx.setLineDash((RELDASH[l.rel] || []).map(v => v * zoom));
    ctx.lineWidth = inSel ? 2 : 1;
    ctx.strokeStyle = inSel
      ? STYLE[NODES[l.s].type].color
      : (sel ? "rgba(32,24,15,0.05)" : "rgba(32,24,15,0.16)");
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }}
  ctx.setLineDash([]);
  for (let i = 0; i < N; i++) {{
    const n = NODES[i];
    const [x, y] = toScreen(n);
    const dimmed = sel && !sel.nodes.has(i);
    ctx.globalAlpha = dimmed ? 0.18 : 1;
    // 2px surface ring separates overlapping marks (the spacer rule).
    ctx.beginPath(); ctx.arc(x, y, n.r * zoom + 2, 0, 7);
    ctx.fillStyle = SURFACE; ctx.fill();
    glyph(ctx, x, y, n.r * zoom, n.type, true);
    if (i === selected || i === hovered) {{
      ctx.beginPath(); ctx.arc(x, y, n.r * zoom + 4, 0, 7);
      ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.stroke();
    }}
    ctx.globalAlpha = 1;
    const showLabel = labelled.has(i) || i === hovered
      || (sel && sel.nodes.has(i));
    if (showLabel && !dimmed && zoom > 0.45) {{
      ctx.font = (n.type === "company" ? "600 " : "") + "11px -apple-system, sans-serif";
      ctx.fillStyle = n.type === "company" ? INK : MUTED;
      ctx.textAlign = "center";
      ctx.fillText(n.label.slice(0, 28), x, y + n.r * zoom + 13);
    }}
  }}
}}

function frame() {{ if (alpha > 0.031) {{ tick(); draw(); }} requestAnimationFrame(frame); }}
requestAnimationFrame(frame);

// ---- interaction -----------------------------------------------------------
function hit(mx, my) {{
  const [wx, wy] = toWorld(mx, my);
  for (let i = N - 1; i >= 0; i--) {{
    const n = NODES[i];
    const rr = n.r + 6 / zoom;
    if ((wx - n.x) ** 2 + (wy - n.y) ** 2 < rr * rr) return i;
  }}
  return -1;
}}

// The tracing panel: everything one node connects to, evidence included —
// built with textContent only, because labels and evidence are web text.
function showInfo(i) {{
  info.hidden = false;
  info.replaceChildren();
  const n = NODES[i];
  const h = document.createElement("h4"); h.textContent = n.label;
  const t = document.createElement("div"); t.className = "t";
  t.textContent = STYLE[n.type] ? n.type + " · " + n.degree + " connection"
    + (n.degree === 1 ? "" : "s") : "";
  info.append(h, t);
  for (const li of adj[i].slice(0, 24)) {{
    const l = LINKS[li];
    const other = NODES[l.s === i ? l.d : l.s];
    const line = document.createElement("div"); line.className = "e";
    line.textContent = (l.s === i ? (RELLABEL[l.rel] || l.rel) + " → "
                                  : "← " + (RELLABEL[l.rel] || l.rel) + " — ")
                       + other.label;
    info.append(line);
    if (l.evidence) {{
      const ev = document.createElement("div"); ev.className = "ev";
      ev.textContent = "  " + l.evidence;
      info.append(ev);
    }}
  }}
}}

canvas.addEventListener("pointerdown", e => {{
  const i = hit(e.offsetX, e.offsetY);
  if (i >= 0) {{ dragging = NODES[i]; alpha = Math.max(alpha, 0.3); }}
  else panning = {{ x: e.offsetX - panX, y: e.offsetY - panY }};
  canvas.setPointerCapture(e.pointerId);
}});
canvas.addEventListener("pointermove", e => {{
  if (dragging) {{
    [dragging.x, dragging.y] = toWorld(e.offsetX, e.offsetY);
    alpha = Math.max(alpha, 0.25); draw();
  }} else if (panning) {{
    panX = e.offsetX - panning.x; panY = e.offsetY - panning.y; draw();
  }} else {{
    const i = hit(e.offsetX, e.offsetY);
    if (i !== hovered) {{ hovered = i; canvas.style.cursor = i >= 0 ? "pointer" : ""; draw(); }}
  }}
}});
canvas.addEventListener("pointerup", e => {{
  if (dragging) {{
    const i = hit(e.offsetX, e.offsetY);
    if (i >= 0 && NODES[i] === dragging) {{ selected = i; showInfo(i); }}
  }} else if (panning) {{
    const moved = Math.abs(e.offsetX - (panning.x + panX))
                + Math.abs(e.offsetY - (panning.y + panY));
    if (moved < 3) {{ selected = -1; info.hidden = true; }}
  }}
  dragging = null; panning = null; draw();
}});
canvas.addEventListener("dblclick", () => {{ selected = -1; info.hidden = true; draw(); }});
canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0012);
  const nz = Math.min(Math.max(zoom * f, 0.25), 4);
  panX = e.offsetX - (e.offsetX - panX) * nz / zoom;
  panY = e.offsetY - (e.offsetY - panY) * nz / zoom;
  zoom = nz; draw();
}}, {{ passive: false }});

// A selected focus opens pre-traced.
if (FOCUS) NODES.forEach((n, i) => {{
  if (n.key === FOCUS) {{ selected = i; showInfo(i); }}
}});
draw();
</script>
"""
