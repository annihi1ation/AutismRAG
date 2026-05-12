"""
Convert unified_kg.json to an interactive HTML visualization.
Preserves entity types (colour-coded) and relation labels.

Usage:
    python unified_kg_to_html.py [--input <path>] [--output <path>]
"""

import json
import argparse
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Unified Knowledge Graph</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; overflow: hidden; }}

    /* ── Toolbar ────────────────────────────────── */
    #toolbar {{
      position: fixed; top: 0; left: 0; right: 0; height: 50px;
      background: #16213e; display: flex; align-items: center;
      padding: 0 16px; gap: 12px; z-index: 10;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }}
    #toolbar h1 {{ font-size: 16px; color: #a8d8ea; flex: 0 0 auto; white-space: nowrap; }}
    #search {{
      flex: 1; max-width: 280px; padding: 6px 10px;
      border-radius: 20px; border: 1px solid #334;
      background: #0f3460; color: #e0e0e0; font-size: 13px; outline: none;
    }}
    #search::placeholder {{ color: #778; }}
    #type-filter {{
      padding: 5px 8px; border-radius: 14px; border: 1px solid #334;
      background: #0f3460; color: #e0e0e0; font-size: 12px; outline: none;
      max-width: 180px;
    }}
    .btn {{
      padding: 5px 12px; border-radius: 14px; border: none; cursor: pointer;
      font-size: 12px; font-weight: 600; transition: background 0.2s;
    }}
    #btn-reset {{ background: #e94560; color: #fff; }}
    #btn-reset:hover {{ background: #c73652; }}
    #btn-labels {{ background: #0f3460; color: #a8d8ea; border: 1px solid #334; }}
    #btn-labels:hover {{ background: #1a4a7a; }}
    #stats {{ font-size: 12px; color: #778; margin-left: auto; white-space: nowrap; }}

    /* ── Graph container ───────────────────────── */
    #container {{ position: fixed; top: 50px; left: 0; right: 0; bottom: 0; }}
    svg {{ width: 100%; height: 100%; }}

    .node circle {{
      stroke-width: 2px;
      cursor: pointer;
      transition: r 0.15s;
    }}
    .node text {{
      font-size: 11px;
      fill: #ddd;
      pointer-events: none;
      text-shadow: 0 1px 3px #000;
    }}
    .link {{
      stroke-opacity: 0.45;
      stroke-width: 1.2px;
    }}
    .link-label {{
      font-size: 8px;
      fill: #889;
      pointer-events: none;
    }}

    /* ── Tooltip ────────────────────────────────── */
    #tooltip {{
      position: fixed; background: #0f3460; border: 1px solid #334;
      border-radius: 8px; padding: 10px 14px; font-size: 12px;
      pointer-events: none; opacity: 0; transition: opacity 0.15s;
      max-width: 340px; line-height: 1.6; z-index: 20;
    }}
    #tooltip strong {{ color: #a8d8ea; }}
    #tooltip .tt-type {{
      display: inline-block; padding: 1px 7px; border-radius: 8px;
      font-size: 10px; font-weight: 600; margin-left: 4px;
    }}
    #tooltip .tt-aliases {{ color: #aab; font-size: 11px; margin-top: 4px; }}
    #tooltip .tt-section {{ color: #889; font-size: 11px; }}

    /* ── Legend ──────────────────────────────────── */
    #legend {{
      position: fixed; bottom: 16px; right: 16px;
      background: rgba(15,52,96,0.92); border-radius: 10px;
      padding: 12px 16px; font-size: 11px; line-height: 1.9;
      max-height: 60vh; overflow-y: auto; z-index: 15;
      border: 1px solid #334;
    }}
    #legend h4 {{ font-size: 13px; color: #a8d8ea; margin-bottom: 6px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; cursor: pointer; }}
    .legend-item:hover {{ color: #fff; }}
    .legend-dot {{
      display: inline-block; width: 12px; height: 12px;
      border-radius: 50%; flex-shrink: 0;
    }}
    .legend-count {{ color: #667; font-size: 10px; }}
  </style>
</head>
<body>
<div id="toolbar">
  <h1>Unified Knowledge Graph</h1>
  <input id="search" type="text" placeholder="Search entity…" />
  <select id="type-filter"><option value="">All types</option></select>
  <button class="btn" id="btn-labels">Labels</button>
  <button class="btn" id="btn-reset">Reset</button>
  <span id="stats"></span>
</div>
<div id="container"><svg id="svg"></svg></div>
<div id="tooltip"></div>
<div id="legend"><h4>Entity Types</h4><div id="legend-items"></div></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
/* ── Data ──────────────────────────────────────── */
const graphData = {graph_data};

/* ── Colour palette for 21 entity types ────────── */
const TYPE_COLORS = {{
  Assessment_Tool:        '#4fc3f7',
  Therapeutic_Approach:   '#81c784',
  Functional_Ability:     '#ffb74d',
  Symptom_Behavior:       '#e57373',
  Disorder:               '#ba68c8',
  Social_Environment:     '#4db6ac',
  Other:                  '#90a4ae',
  Healthcare_Access:      '#aed581',
  Person_Role:            '#f06292',
  Institution_System:     '#7986cb',
  Neurobiology:           '#ff8a65',
  Symptom_Emotion:        '#ef5350',
  Medication:             '#26c6da',
  Clinical_Severity:      '#ffd54f',
  Developmental_Stage:    '#a1887f',
  Educational_Approach:   '#dce775',
  Diagnostic_Framework:   '#9575cd',
  Socioeconomic:          '#4dd0e1',
  Trauma:                 '#ff7043',
  Intersectional_Identity:'#ce93d8',
  Gene:                   '#80cbc4',
}};

const fallbackColor = '#607d8b';
function typeColor(t) {{ return TYPE_COLORS[t] || fallbackColor; }}

/* ── Build entity lookup ───────────────────────── */
const entityLookup = new Map();
graphData.entities.forEach(e => entityLookup.set(e.canonical_name, e));

/* ── Build nodes & links ───────────────────────── */
const nodeMap = new Map();
graphData.triples.forEach(t => {{
  if (!nodeMap.has(t.head)) {{
    const ent = entityLookup.get(t.head);
    nodeMap.set(t.head, {{
      id: t.head,
      type: ent ? ent.entity_type : 'Other',
      aliases: ent ? ent.aliases : [],
      sources: ent ? ent.source_articles.length : 0,
      degree: 0
    }});
  }}
  if (!nodeMap.has(t.tail)) {{
    const ent = entityLookup.get(t.tail);
    nodeMap.set(t.tail, {{
      id: t.tail,
      type: ent ? ent.entity_type : 'Other',
      aliases: ent ? ent.aliases : [],
      sources: ent ? ent.source_articles.length : 0,
      degree: 0
    }});
  }}
  nodeMap.get(t.head).degree++;
  nodeMap.get(t.tail).degree++;
}});

const nodes = Array.from(nodeMap.values());
const links = graphData.triples.map(t => ({{
  source: t.head,
  target: t.tail,
  relation: t.relation,
  evidence: t.evidence_count || 1,
  confidence: t.confidence || 0,
  conflict: t.conflict_status || 'consistent'
}}));

/* ── Stats ─────────────────────────────────────── */
document.getElementById('stats').textContent =
  `${{nodes.length}} entities  ·  ${{links.length}} triples  ·  ${{graphData.metadata ? graphData.metadata.source_articles : '?'}} articles`;

/* ── Type counts for legend & filter ───────────── */
const typeCounts = {{}};
nodes.forEach(n => {{ typeCounts[n.type] = (typeCounts[n.type] || 0) + 1; }});
const typesSorted = Object.entries(typeCounts).sort((a,b) => b[1] - a[1]);

// Populate legend
const legendDiv = document.getElementById('legend-items');
typesSorted.forEach(([t, c]) => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.dataset.type = t;
  item.innerHTML = `<span class="legend-dot" style="background:${{typeColor(t)}}"></span>${{t.replace(/_/g,' ')}} <span class="legend-count">(${{c}})</span>`;
  item.addEventListener('click', () => filterByType(t));
  legendDiv.appendChild(item);
}});

// Populate type filter dropdown
const typeFilter = document.getElementById('type-filter');
typesSorted.forEach(([t, c]) => {{
  const opt = document.createElement('option');
  opt.value = t;
  opt.textContent = `${{t.replace(/_/g,' ')}} (${{c}})`;
  typeFilter.appendChild(opt);
}});

/* ── SVG setup ─────────────────────────────────── */
const svg = d3.select('#svg');
const container = document.getElementById('container');
function getSize() {{ return {{ w: container.clientWidth, h: container.clientHeight }}; }}
let {{ w, h }} = getSize();
const g = svg.append('g');

// Zoom
const zoom = d3.zoom().scaleExtent([0.05, 10])
  .on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

// Arrow markers per relation colour
svg.append('defs').append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', '0 -5 10 10')
  .attr('refX', 22).attr('refY', 0)
  .attr('markerWidth', 5).attr('markerHeight', 5)
  .attr('orient', 'auto')
  .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#556');

/* ── Relation colour scale ─────────────────────── */
const relationTypes = [...new Set(links.map(l => l.relation))].sort();
const relColorScale = d3.scaleOrdinal(d3.schemeTableau10).domain(relationTypes);

/* ── Simulation ────────────────────────────────── */
const simulation = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d => d.id).distance(100))
  .force('charge', d3.forceManyBody().strength(-200))
  .force('center', d3.forceCenter(w / 2, h / 2))
  .force('collision', d3.forceCollide(25))
  .force('x', d3.forceX(w / 2).strength(0.03))
  .force('y', d3.forceY(h / 2).strength(0.03));

/* ── Draw links ────────────────────────────────── */
const link = g.append('g').selectAll('line')
  .data(links).join('line')
  .attr('class', 'link')
  .attr('stroke', d => relColorScale(d.relation))
  .attr('stroke-width', d => 0.8 + Math.min(d.evidence, 6) * 0.4)
  .attr('marker-end', 'url(#arrow)');

/* ── Draw link labels (hidden by default) ──────── */
let showLabels = false;
const linkLabel = g.append('g').selectAll('text')
  .data(links).join('text')
  .attr('class', 'link-label')
  .attr('display', 'none')
  .text(d => d.relation.length > 20 ? d.relation.slice(0, 19) + '…' : d.relation);

/* ── Draw nodes ────────────────────────────────── */
const node = g.append('g').selectAll('g')
  .data(nodes).join('g')
  .attr('class', 'node')
  .call(d3.drag()
    .on('start', (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on('drag',  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on('end',   (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

node.append('circle')
  .attr('r', d => 6 + Math.sqrt(d.degree) * 2)
  .attr('fill', d => typeColor(d.type))
  .attr('stroke', d => d3.color(typeColor(d.type)).darker(0.8))
  .attr('stroke-width', 1.5);

node.append('text')
  .attr('dy', d => -(8 + Math.sqrt(d.degree) * 2))
  .attr('text-anchor', 'middle')
  .text(d => d.id.length > 24 ? d.id.slice(0, 23) + '…' : d.id);

/* ── Tooltip ───────────────────────────────────── */
const tooltip = document.getElementById('tooltip');

node.on('mouseover', (e, d) => {{
  const out = links.filter(l => (l.source.id || l.source) === d.id);
  const inc = links.filter(l => (l.target.id || l.target) === d.id);
  const outRels = out.map(l => `→ ${{l.relation}} → ${{l.target.id || l.target}}`).slice(0, 5);
  const incRels = inc.map(l => `${{l.source.id || l.source}} → ${{l.relation}} →`).slice(0, 5);
  const aliasText = d.aliases.length > 0
    ? `<div class="tt-aliases">Aliases: ${{d.aliases.slice(0, 6).join(', ')}}${{d.aliases.length > 6 ? ' …+' + (d.aliases.length - 6) : ''}}</div>`
    : '';
  tooltip.innerHTML =
    `<strong>${{d.id}}</strong>` +
    `<span class="tt-type" style="background:${{typeColor(d.type)}}; color:#111">${{d.type.replace(/_/g,' ')}}</span><br>` +
    `<span class="tt-section">Degree: ${{d.degree}} &nbsp;|&nbsp; Sources: ${{d.sources}} articles</span>` +
    aliasText +
    (outRels.length ? `<div class="tt-section" style="margin-top:4px">Out: ${{outRels.join('<br>&nbsp;&nbsp;&nbsp;')}}</div>` : '') +
    (incRels.length ? `<div class="tt-section">In: ${{incRels.join('<br>&nbsp;&nbsp;&nbsp;')}}</div>` : '');
  tooltip.style.opacity = 1;
}})
.on('mousemove', e => {{
  tooltip.style.left = Math.min(e.clientX + 14, window.innerWidth - 360) + 'px';
  tooltip.style.top  = Math.min(e.clientY - 28, window.innerHeight - 200) + 'px';
}})
.on('mouseleave', () => {{ tooltip.style.opacity = 0; }});

/* ── Click to highlight neighbourhood ──────────── */
let selected = null;
node.on('click', (e, d) => {{
  e.stopPropagation();
  if (selected === d.id) {{ selected = null; resetHighlight(); }}
  else {{ selected = d.id; highlight(d.id); }}
}});
svg.on('click', () => {{ selected = null; resetHighlight(); }});

function highlight(id) {{
  const neighbours = new Set([id]);
  links.forEach(l => {{
    const s = l.source.id || l.source, t = l.target.id || l.target;
    if (s === id) neighbours.add(t);
    if (t === id) neighbours.add(s);
  }});
  node.select('circle')
    .attr('fill', d => {{
      if (d.id === id) return '#fff';
      if (neighbours.has(d.id)) return typeColor(d.type);
      return '#2a2a3e';
    }})
    .attr('stroke',  d => d.id === id ? '#ff5252' : (neighbours.has(d.id) ? d3.color(typeColor(d.type)).brighter(0.5) : '#333'))
    .attr('stroke-width', d => d.id === id ? 3 : 1.5);
  link.attr('stroke', l => {{
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return (s === id || t === id) ? relColorScale(l.relation) : '#1e2a40';
  }}).attr('stroke-opacity', l => {{
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return (s === id || t === id) ? 0.9 : 0.08;
  }});
  node.select('text')
    .attr('fill', d => neighbours.has(d.id) ? '#fff' : '#333')
    .attr('font-weight', d => d.id === id ? '700' : '400');
  // Show link labels for selected neighbourhood
  linkLabel.attr('display', l => {{
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return (s === id || t === id) ? null : 'none';
  }});
}}

function resetHighlight() {{
  node.select('circle')
    .attr('fill', d => typeColor(d.type))
    .attr('stroke', d => d3.color(typeColor(d.type)).darker(0.8))
    .attr('stroke-width', 1.5);
  link.attr('stroke', d => relColorScale(d.relation)).attr('stroke-opacity', 0.45);
  node.select('text').attr('fill', '#ddd').attr('font-weight', '400');
  linkLabel.attr('display', showLabels ? null : 'none');
}}

/* ── Filter by type ────────────────────────────── */
let activeTypeFilter = '';
function filterByType(type) {{
  if (activeTypeFilter === type) {{
    activeTypeFilter = '';
    typeFilter.value = '';
  }} else {{
    activeTypeFilter = type;
    typeFilter.value = type;
  }}
  applyTypeFilter();
}}

typeFilter.addEventListener('change', function() {{
  activeTypeFilter = this.value;
  applyTypeFilter();
}});

function applyTypeFilter() {{
  if (!activeTypeFilter) {{
    node.attr('display', null);
    link.attr('display', null);
    linkLabel.attr('display', showLabels ? null : 'none');
    return;
  }}
  const visible = new Set();
  nodes.forEach(n => {{ if (n.type === activeTypeFilter) visible.add(n.id); }});
  // Also show direct neighbours
  links.forEach(l => {{
    const s = l.source.id || l.source, t = l.target.id || l.target;
    if (visible.has(s)) visible.add(t);
    if (visible.has(t)) visible.add(s);
  }});
  node.attr('display', d => visible.has(d.id) ? null : 'none');
  link.attr('display', l => {{
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return (visible.has(s) && visible.has(t)) ? null : 'none';
  }});
  linkLabel.attr('display', l => {{
    if (!showLabels) return 'none';
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return (visible.has(s) && visible.has(t)) ? null : 'none';
  }});
}}

/* ── Toggle edge labels ────────────────────────── */
document.getElementById('btn-labels').addEventListener('click', () => {{
  showLabels = !showLabels;
  document.getElementById('btn-labels').textContent = showLabels ? 'Hide Labels' : 'Labels';
  linkLabel.attr('display', showLabels ? null : 'none');
}});

/* ── Search ────────────────────────────────────── */
document.getElementById('search').addEventListener('input', function() {{
  const q = this.value.trim().toLowerCase();
  if (!q) {{ resetHighlight(); return; }}
  // Search in names and aliases
  const match = nodes.find(n =>
    n.id.toLowerCase().includes(q) ||
    n.aliases.some(a => a.toLowerCase().includes(q))
  );
  if (match) highlight(match.id);
}});

/* ── Reset ─────────────────────────────────────── */
document.getElementById('btn-reset').addEventListener('click', () => {{
  svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity);
  activeTypeFilter = '';
  typeFilter.value = '';
  applyTypeFilter();
  resetHighlight();
  document.getElementById('search').value = '';
}});

/* ── Tick ──────────────────────────────────────── */
simulation.on('tick', () => {{
  link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  linkLabel
    .attr('x', d => (d.source.x + d.target.x) / 2)
    .attr('y', d => (d.source.y + d.target.y) / 2);
  node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});

/* ── Resize ────────────────────────────────────── */
window.addEventListener('resize', () => {{
  const s = getSize();
  simulation.force('center', d3.forceCenter(s.w / 2, s.h / 2))
    .force('x', d3.forceX(s.w / 2).strength(0.03))
    .force('y', d3.forceY(s.h / 2).strength(0.03))
    .alpha(0.2).restart();
}});
</script>
</body>
</html>
"""


def load_graph(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph_data(kg: dict) -> dict:
    """Build a slim graph payload preserving entity types and relation info."""
    entities = [
        {
            "canonical_name": e["canonical_name"],
            "entity_type": e.get("entity_type", "Other"),
            "aliases": e.get("aliases", []),
            "source_articles": e.get("source_articles", []),
        }
        for e in kg.get("entities", [])
    ]
    triples = [
        {
            "head": t["head"],
            "relation": t["relation"],
            "tail": t["tail"],
            "confidence": round(t.get("confidence", 0), 3),
            "evidence_count": t.get("evidence_count", 1),
            "conflict_status": t.get("conflict_status", "consistent"),
        }
        for t in kg.get("triples", [])
    ]
    metadata = kg.get("metadata", {})
    return {"entities": entities, "triples": triples, "metadata": metadata}


def convert(input_path: Path, output_path: Path) -> None:
    kg = load_graph(input_path)
    graph_data = build_graph_data(kg)
    graph_json = json.dumps(graph_data, ensure_ascii=False)

    html = HTML_TEMPLATE.format(graph_data=graph_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    n_heads = set(t["head"] for t in graph_data["triples"])
    n_tails = set(t["tail"] for t in graph_data["triples"])
    print(f"[OK] HTML written to: {output_path}")
    print(f"     Entities : {len(graph_data['entities'])}")
    print(f"     Nodes    : {len(n_heads | n_tails)}")
    print(f"     Edges    : {len(graph_data['triples'])}")
    rels = set(t["relation"] for t in graph_data["triples"])
    print(f"     Relations: {len(rels)} ({', '.join(sorted(rels))})")


def _default_input_path() -> Path:
    output_root = Path(__file__).resolve().parent.parent / "output"
    nested_path = output_root / "unified_kg" / "unified_kg.json"
    legacy_path = output_root / "unified_kg.json"
    return nested_path if nested_path.exists() else legacy_path


def main():
    parser = argparse.ArgumentParser(description="Convert unified_kg.json to HTML.")
    parser.add_argument(
        "--input", "-i",
        default=str(_default_input_path()),
        help="Path to unified_kg.json",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).resolve().parent / "unified_kg.html"),
        help="Path for output HTML file",
    )
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
