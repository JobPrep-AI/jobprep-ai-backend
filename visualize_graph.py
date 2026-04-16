import pandas as pd
import networkx as nx
import community as community_louvain
from collections import defaultdict
from pyvis.network import Network
from snowflake_connection import get_snowflake_connection
import math

conn = get_snowflake_connection()

# ── Rendering/perf knobs (tune as needed) ──────────────────────────────────
# Keep the source graph reasonably sized before clustering.
MAX_SOURCE_ROWS = 12000
TOP_COMPANIES = 120
MAX_QUESTIONS_PER_COMPANY_ROLE = 20

# Big speed improvement: hide individual question nodes in visualization.
# (Graph still uses questions internally for clustering.)
RENDER_QUESTION_NODES = False
# Stop node movement after initial layout settles.
FREEZE_AFTER_STABILIZATION = True


def fetch_pandas_df(query):
    cur = conn.cursor()
    try:
        cur.execute(query)
        df = cur.fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df
    finally:
        cur.close()


# ── Node colors per type ──────────────────────────────────────────────────
COLORS = {
    "company":  "#534AB7",   # purple
    "role":     "#1D9E75",   # teal
    "question": "#EF9F27",   # amber
    "category": "#D85A30",   # coral
    "cluster":  "#378ADD",   # blue
}

# ── Node sizes per type ───────────────────────────────────────────────────
SIZES = {
    "company":  35,
    "role":     25,
    "category": 20,
    "cluster":  30,
    "question": 12,
}


print("Loading question bank from Snowflake...")
jobs_df = fetch_pandas_df("""
    SELECT company_name, role_name, interview_question, question_category_enhanced
    FROM JOBPREP_DB.MARTS.MART_QUESTION_BANK
""")

jobs_df = jobs_df.drop_duplicates(
    subset=["company_name", "role_name", "interview_question"]
)
jobs_df = jobs_df.dropna(
    subset=["company_name", "role_name", "interview_question", "question_category_enhanced"]
)
jobs_df = jobs_df[jobs_df["interview_question"].str.len() > 15]

# Trim source volume to keep interactive HTML responsive.
top_companies = jobs_df["company_name"].value_counts().head(TOP_COMPANIES).index
jobs_df = jobs_df[jobs_df["company_name"].isin(top_companies)]
jobs_df = (
    jobs_df.groupby(["company_name", "role_name"], group_keys=False)
    .head(MAX_QUESTIONS_PER_COMPANY_ROLE)
)
if len(jobs_df) > MAX_SOURCE_ROWS:
    jobs_df = jobs_df.sample(n=MAX_SOURCE_ROWS, random_state=42)

print(f"Loaded {len(jobs_df)} questions across "
      f"{jobs_df['company_name'].nunique()} companies.")


# ── Rebuild the knowledge graph (same logic as build_graph_index.py) ──────
print("Building knowledge graph...")

G = nx.Graph()

for _, row in jobs_df.iterrows():
    company  = row["company_name"]
    role     = row["role_name"]
    question = row["interview_question"]
    category = row["question_category_enhanced"]

    G.add_node(company,  type="company")
    G.add_node(role,     type="role")
    G.add_node(question, type="question")
    G.add_node(category, type="category")

    G.add_edge(company,  role,     relation="hires_for")
    G.add_edge(role,     question, relation="asks")
    G.add_edge(question, category, relation="belongs_to")

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ── Louvain community detection ───────────────────────────────────────────
print("Detecting communities...")
partition = community_louvain.best_partition(G)

# Map cluster id → label using the most common category in that cluster
cluster_categories = defaultdict(list)
for node, cid in partition.items():
    if G.nodes[node].get("type") == "category":
        cluster_categories[cid].append(node)

cluster_labels = {}
for cid, cats in cluster_categories.items():
    label = max(set(cats), key=cats.count) if cats else f"Cluster {cid}"
    cluster_labels[cid] = f"Cluster {cid}: {label}"

# Add cluster nodes and edges from each category
for node, cid in partition.items():
    if G.nodes[node].get("type") == "category":
        cluster_node = cluster_labels.get(cid, f"Cluster {cid}")
        G.add_node(cluster_node, type="cluster")
        G.add_edge(node, cluster_node, relation="in_cluster")

print(f"Graph after clusters: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ── Build PyVis network ───────────────────────────────────────────────────
print("Building visualization...")

net = Network(
    height="800px",
    width="100%",
    bgcolor="#ffffff",
    font_color="#333333",
    notebook=False,
    directed=False,
    cdn_resources="in_line",
)

# Physics: Barnes-Hut gives the best spread for large graphs
net.barnes_hut(
    gravity=-8000,
    central_gravity=0.3,
    spring_length=120,
    spring_strength=0.04,
    damping=0.09,
)
net.set_options("""
{
  "interaction": {
    "hover": true,
    "hideEdgesOnDrag": true,
    "tooltipDelay": 120
  },
  "physics": {
    "enabled": true,
    "minVelocity": 0.15,
    "stabilization": {
      "enabled": true,
      "iterations": 250,
      "updateInterval": 25
    }
  }
}
""")

# Add nodes
for node, attrs in G.nodes(data=True):
    ntype = attrs.get("type", "question")
    if not RENDER_QUESTION_NODES and ntype == "question":
        continue

    color = COLORS.get(ntype, "#888888")
    size = SIZES.get(ntype, 12)
    label = node if len(str(node)) <= 50 else str(node)[:48] + "…"
    title = f"<b>{ntype.upper()}</b><br>{node}"

    net.add_node(
        node,
        label=label if ntype != "question" else "",
        title=title,
        color=color,
        size=size,
        font={"size": 11 if ntype in ("company", "cluster") else 9},
        borderWidth=2,
        borderWidthSelected=4,
    )

# Add edges
EDGE_COLORS = {
    "hires_for":  "#534AB7",
    "asks":       "#EF9F27",
    "belongs_to": "#D85A30",
    "in_cluster": "#378ADD",
}

if RENDER_QUESTION_NODES:
    for u, v, attrs in G.edges(data=True):
        rel = attrs.get("relation", "")
        color = EDGE_COLORS.get(rel, "#cccccc")
        dashes = rel == "in_cluster"

        net.add_edge(
            u,
            v,
            title=rel,
            color={"color": color, "opacity": 0.4},
            width=1.5 if rel in ("hires_for", "in_cluster") else 0.8,
            dashes=dashes,
        )
else:
    # Collapsed view: company->role, role->category, category->cluster
    company_role_counts = jobs_df.groupby(["company_name", "role_name"]).size().reset_index(name="cnt")
    for row in company_role_counts.itertuples(index=False):
        width = 1.0 + math.log1p(row.cnt)
        net.add_edge(
            row.company_name,
            row.role_name,
            title=f"hires_for ({row.cnt})",
            color={"color": EDGE_COLORS["hires_for"], "opacity": 0.45},
            width=width,
        )

    role_cat_counts = jobs_df.groupby(["role_name", "question_category_enhanced"]).size().reset_index(name="cnt")
    for row in role_cat_counts.itertuples(index=False):
        width = 0.8 + math.log1p(row.cnt)
        net.add_edge(
            row.role_name,
            row.question_category_enhanced,
            title=f"asks/belongs_to ({row.cnt})",
            color={"color": EDGE_COLORS["asks"], "opacity": 0.35},
            width=width,
        )

    for u, v, attrs in G.edges(data=True):
        rel = attrs.get("relation", "")
        if rel != "in_cluster":
            continue
        if G.nodes[u].get("type") != "category":
            continue
        net.add_edge(
            u,
            v,
            title="in_cluster",
            color={"color": EDGE_COLORS["in_cluster"], "opacity": 0.35},
            width=1.2,
            dashes=True,
        )


# ── Legend HTML injected into the page ───────────────────────────────────
legend_html = """
<div style="position:fixed;top:16px;left:16px;background:#fff;border:1px solid #ddd;
            border-radius:10px;padding:14px 18px;font-family:sans-serif;font-size:12px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);z-index:999;">
  <b style="font-size:13px">Node types</b>
  <div style="margin-top:8px;display:flex;flex-direction:column;gap:5px">
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#534AB7;margin-right:6px"></span>Company</div>
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#1D9E75;margin-right:6px"></span>Role</div>
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#EF9F27;margin-right:6px"></span>Question</div>
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#D85A30;margin-right:6px"></span>Category</div>
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#378ADD;margin-right:6px"></span>Cluster</div>
  </div>
  <b style="font-size:13px;display:block;margin-top:10px">Edge relations</b>
  <div style="margin-top:8px;display:flex;flex-direction:column;gap:5px">
    <div><span style="display:inline-block;width:20px;height:2px;background:#534AB7;margin-right:6px;vertical-align:middle"></span>hires_for</div>
    <div><span style="display:inline-block;width:20px;height:2px;background:#EF9F27;margin-right:6px;vertical-align:middle"></span>asks</div>
    <div><span style="display:inline-block;width:20px;height:2px;background:#D85A30;margin-right:6px;vertical-align:middle"></span>belongs_to</div>
    <div><span style="display:inline-block;width:20px;height:2px;background:#378ADD;margin-right:6px;vertical-align:middle;border-top:2px dashed #378ADD"></span>in_cluster</div>
  </div>
  <div style="margin-top:10px;color:#888;font-size:11px">
    Scroll to zoom · Drag to pan<br>Click node to highlight
  </div>
</div>
"""

# ── Save ──────────────────────────────────────────────────────────────────
output_path = "jobprep_knowledge_graph.html"
# Generate HTML first and write with UTF-8 explicitly
# (Windows default cp1252 can fail on Unicode characters).
html = net.generate_html(notebook=False)

if FREEZE_AFTER_STABILIZATION:
    freeze_script = """
<script type="text/javascript">
if (typeof network !== "undefined") {
  network.once("stabilizationIterationsDone", function () {
    network.setOptions({ physics: false });
  });
}
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", freeze_script + "</body>")

if "</body>" in html:
    html = html.replace("</body>", legend_html + "</body>")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nSaved → {output_path}")
print("Open it in your browser to explore the graph interactively.")