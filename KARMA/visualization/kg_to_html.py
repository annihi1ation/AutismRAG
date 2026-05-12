"""
Convert knowledge_graph.json to an interactive HTML visualization.
Only uses head, relation, and tail from each triple.

Usage:
    python kg_to_html.py [--input <path>] [--output <path>]
"""

import json
import argparse
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Knowledge Graph</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; overflow: hidden; }}

    #toolbar {{
      position: fixed; top: 0; left: 0; right: 0; height: 50px;
      background: #16213e; display: flex; align-items: center;
      padding: 0 16px; gap: 12px; z-index: 10;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }}
    #toolbar h1 {{ font-size: 16px; color: #a8d8ea; flex: 0 0 auto; }}
    #search {{
      flex: 1; max-width: 300px; padding: 6px 10px;
      border-radius: 20px; border: 1px solid #334;
      background: #0f3460; color: #e0e0e0; font-size: 13px; outline: none;
    }}
    #search::placeholder {{ color: #778; }}
    .btn {{
      padding: 5px 12px; border-radius: 14px; border: none; cursor: pointer;
      font-size: 12px; font-weight: 600; transition: background 0.2s;
    }}
    #btn-reset {{ background: #e94560; color: #fff; }}
    #btn-reset:hover {{ background: #c73652; }}
    #stats {{ font-size: 12px; color: #778; margin-left: auto; }}

    #container {{ position: fixed; top: 50px; left:0; right:0; bottom:0; }}
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
      stroke-opacity: 0.55;
      stroke-width: 1.5px;
    }}
    .link-label {{
      font-size: 9px;
      fill: #aaa;
      pointer-events: none;
    }}

    /* Tooltip */
    #tooltip {{
      position: fixed; background: #0f3460; border: 1px solid #334;
      border-radius: 8px; padding: 10px 14px; font-size: 12px;
      pointer-events: none; opacity: 0; transition: opacity 0.2s;
      max-width: 280px; line-height: 1.6; z-index: 20;
    }}
    #tooltip strong {{ color: #a8d8ea; }}

    /* Legend */
    #legend {{
      position: fixed; bottom: 16px; right: 16px;
      background: rgba(15,52,96,0.85); border-radius: 8px;
      padding: 10px 14px; font-size: 11px; line-height: 1.8;
    }}
    #legend h4 {{ font-size: 12px; color: #a8d8ea; margin-bottom: 4px; }}
    .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
  </style>
</head>
<body>
<div id="toolbar">
  <h1>Knowledge Graph</h1>
  <input id="search" type="text" placeholder="Search node…" />
  <button class="btn" id="btn-reset">Reset View</button>
  <span id="stats"></span>
</div>
<div id="container"><svg id="svg"></svg></div>
<div id="tooltip"></div>
<div id="legend">
  <h4>Legend</h4>
  <div><span class="legend-dot" style="background:#4fc3f7"></span>Hub node (many connections)</div>
  <div><span class="legend-dot" style="background:#81c784"></span>Regular node</div>
  <div><span class="legend-dot" style="background:#e57373"></span>Highlighted / selected</div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const graphData = {graph_data};

// Build nodes & links
const nodeMap = new Map();
graphData.triples.forEach(t => {{
  if (!nodeMap.has(t.head)) nodeMap.set(t.head, {{ id: t.head, degree: 0 }});
  if (!nodeMap.has(t.tail)) nodeMap.set(t.tail, {{ id: t.tail, degree: 0 }});
  nodeMap.get(t.head).degree++;
  nodeMap.get(t.tail).degree++;
}});

const nodes = Array.from(nodeMap.values());
const links = graphData.triples.map(t => ({{
  source: t.head,
  target: t.tail,
  relation: t.relation
}}));

document.getElementById('stats').textContent =
  `Nodes: ${{nodes.length}}  |  Edges: ${{links.length}}`;

// Colour scale by degree
const maxDeg = d3.max(nodes, d => d.degree);
const colorScale = d3.scaleLinear()
  .domain([1, Math.max(2, maxDeg)])
  .range(['#81c784', '#4fc3f7']);

const svg = d3.select('#svg');
const container = document.getElementById('container');

function getSize() {{
  return {{ w: container.clientWidth, h: container.clientHeight }};
}}

let {{ w, h }} = getSize();

const g = svg.append('g');

// Zoom
const zoom = d3.zoom()
  .scaleExtent([0.1, 8])
  .on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

// Arrow marker
svg.append('defs').append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', '0 -5 10 10')
  .attr('refX', 22).attr('refY', 0)
  .attr('markerWidth', 6).attr('markerHeight', 6)
  .attr('orient', 'auto')
  .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#556');

// Simulation
const simulation = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d => d.id).distance(120))
  .force('charge', d3.forceManyBody().strength(-300))
  .force('center', d3.forceCenter(w / 2, h / 2))
  .force('collision', d3.forceCollide(30));

// Links
const link = g.append('g').selectAll('line')
  .data(links).join('line')
  .attr('class', 'link')
  .attr('stroke', '#3a4a6b')
  .attr('marker-end', 'url(#arrow)');

// Link labels
const linkLabel = g.append('g').selectAll('text')
  .data(links).join('text')
  .attr('class', 'link-label')
  .text(d => d.relation.length > 25 ? d.relation.slice(0, 24) + '…' : d.relation);

// Nodes
const node = g.append('g').selectAll('g')
  .data(nodes).join('g')
  .attr('class', 'node')
  .call(d3.drag()
    .on('start', (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on('drag',  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on('end',   (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

node.append('circle')
  .attr('r', d => 7 + Math.sqrt(d.degree) * 2.5)
  .attr('fill', d => colorScale(d.degree))
  .attr('stroke', '#1a1a2e');

node.append('text')
  .attr('dy', d => -(9 + Math.sqrt(d.degree) * 2.5))
  .attr('text-anchor', 'middle')
  .text(d => d.id.length > 22 ? d.id.slice(0, 21) + '…' : d.id);

// Tooltip
const tooltip = document.getElementById('tooltip');

node.on('mouseover', (e, d) => {{
  const relatedLinks = links.filter(l =>
    (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id
  );
  const outgoing = relatedLinks.filter(l => (l.source.id || l.source) === d.id);
  const incoming = relatedLinks.filter(l => (l.target.id || l.target) === d.id);
  tooltip.innerHTML =
    `<strong>${{d.id}}</strong><br>` +
    `Degree: ${{d.degree}}<br>` +
    `Outgoing: ${{outgoing.length}} &nbsp; Incoming: ${{incoming.length}}`;
  tooltip.style.opacity = 1;
}})
.on('mousemove', e => {{
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY - 28) + 'px';
}})
.on('mouseleave', () => {{ tooltip.style.opacity = 0; }});

// Click to highlight neighbourhood
let selected = null;
node.on('click', (e, d) => {{
  e.stopPropagation();
  if (selected === d.id) {{
    selected = null; resetHighlight();
  }} else {{
    selected = d.id; highlight(d.id);
  }}
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
    .attr('fill', d => neighbours.has(d.id) ? (d.id === id ? '#e57373' : '#ffb74d') : '#2a3a5e')
    .attr('stroke', d => d.id === id ? '#ff5252' : '#1a1a2e');
  link.attr('stroke', l => {{
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return (s === id || t === id) ? '#e57373' : '#1e2a40';
  }}).attr('stroke-opacity', l => {{
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return (s === id || t === id) ? 0.9 : 0.15;
  }});
  node.select('text').attr('fill', d => neighbours.has(d.id) ? '#fff' : '#445');
}}

function resetHighlight() {{
  node.select('circle')
    .attr('fill', d => colorScale(d.degree))
    .attr('stroke', '#1a1a2e');
  link.attr('stroke', '#3a4a6b').attr('stroke-opacity', 0.55);
  node.select('text').attr('fill', '#ddd');
}}

// Simulation tick
simulation.on('tick', () => {{
  link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  linkLabel
    .attr('x', d => (d.source.x + d.target.x) / 2)
    .attr('y', d => (d.source.y + d.target.y) / 2);
  node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});

// Search
document.getElementById('search').addEventListener('input', function() {{
  const q = this.value.trim().toLowerCase();
  if (!q) {{ resetHighlight(); return; }}
  const match = nodes.find(n => n.id.toLowerCase().includes(q));
  if (match) highlight(match.id);
}});

// Reset view
document.getElementById('btn-reset').addEventListener('click', () => {{
  svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity);
  resetHighlight();
}});

// Resize
window.addEventListener('resize', () => {{
  const s = getSize();
  simulation.force('center', d3.forceCenter(s.w / 2, s.h / 2)).alpha(0.2).restart();
}});
</script>
</body>
</html>
"""


def load_graph(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph_data(kg: dict) -> dict:
    """Extract only head, relation, tail from each triple."""
    triples = [
        {"head": t["head"], "relation": t["relation"], "tail": t["tail"]}
        for t in kg.get("triples", [])
    ]
    return {"triples": triples}


def convert(input_path: Path, output_path: Path) -> None:
    kg = load_graph(input_path)
    graph_data = build_graph_data(kg)
    graph_json = json.dumps(graph_data, ensure_ascii=False, indent=2)

    html = HTML_TEMPLATE.format(graph_data=graph_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] HTML written to: {output_path}")
    print(f"     Nodes : {len(set(t['head'] for t in graph_data['triples']) | set(t['tail'] for t in graph_data['triples']))}")
    print(f"     Edges : {len(graph_data['triples'])}")


def main():
    parser = argparse.ArgumentParser(description="Convert knowledge_graph.json to HTML.")
    parser.add_argument(
        "--input", "-i",
        default=str(Path(__file__).parent.parent / "knowledge_graph.json"),
        help="Path to knowledge_graph.json",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).parent / "knowledge_graph.html"),
        help="Path for output HTML file",
    )
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
