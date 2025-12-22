"""Graph visualization module for reasoning graphs.

This module provides tools to visualize reasoning graphs in various formats,
including DOT/Graphviz, interactive HTML, and static images.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json
from datetime import datetime

# Logger setup
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Visualizer for reasoning graphs.
    
    This class provides methods to export and visualize reasoning graphs
    in multiple formats, with support for highlighting pruned/merged nodes
    and showing path genealogy.
    """
    
    def __init__(self):
        """Initialize the graph visualizer."""
        self.pruned_nodes: Set[int] = set()
        self.merged_nodes: Set[int] = set()
        logger.info("[GraphVisualizer] Initialized graph visualizer")
    
    def mark_pruned(self, node_ids: List[int]) -> None:
        """Mark nodes as pruned for visualization.
        
        Args:
            node_ids: List of node IDs that were pruned
        """
        self.pruned_nodes.update(node_ids)
        logger.debug(f"[GraphVisualizer] Marked {len(node_ids)} nodes as pruned")
    
    def mark_merged(self, node_ids: List[int]) -> None:
        """Mark nodes as merged for visualization.
        
        Args:
            node_ids: List of node IDs that were merged
        """
        self.merged_nodes.update(node_ids)
        logger.debug(f"[GraphVisualizer] Marked {len(node_ids)} nodes as merged")
    
    def export_to_dot(
        self,
        graph: Any,
        output_path: str,
        show_scores: bool = True,
        show_metadata: bool = True,
        highlight_best_path: bool = True
    ) -> None:
        """Export graph to DOT format for Graphviz.
        
        Args:
            graph: ReasoningGraph object to visualize
            output_path: Path to save the DOT file
            show_scores: Whether to display node scores
            show_metadata: Whether to display node metadata
            highlight_best_path: Whether to highlight the best-scoring path
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Find best path if highlighting
            best_path_nodes = set()
            if highlight_best_path and graph.leaf_ids:
                best_leaf = max(
                    graph.leaf_ids,
                    key=lambda nid: graph.get_node(nid).score if graph.get_node(nid) else 0.0
                )
                best_path_nodes = set(graph.get_path(best_leaf))
                logger.debug(f"[GraphVisualizer] Best path: {best_path_nodes}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("digraph ReasoningGraph {\n")
                f.write("  rankdir=TB;\n")
                f.write("  node [shape=box, style=rounded];\n")
                f.write("  graph [fontname=\"Arial\", fontsize=12];\n")
                f.write("  node [fontname=\"Arial\", fontsize=10];\n")
                f.write("  edge [fontname=\"Arial\", fontsize=9];\n\n")
                
                # Add title
                f.write(f'  labelloc="t";\n')
                f.write(f'  label="Reasoning Graph - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}";\n\n')
                
                # Write nodes
                for node_id, node in graph.nodes.items():
                    color = self._get_node_color(node.score, node_id, best_path_nodes)
                    style = self._get_node_style(node_id)
                    
                    # Build label
                    label_parts = [f"Node {node_id}"]
                    if show_scores:
                        label_parts.append(f"score: {node.score:.3f}")
                    
                    if show_metadata and node.metadata:
                        # Show key metadata
                        if 'agent_name' in node.metadata:
                            label_parts.append(f"agent: {node.metadata['agent_name']}")
                        if 'step' in node.metadata:
                            label_parts.append(f"step: {node.metadata['step']}")
                        if 'merged_from' in node.metadata:
                            label_parts.append(f"merged from: {node.metadata['merged_from']}")
                        if 'branched_from' in node.metadata:
                            label_parts.append(f"branched from: {node.metadata['branched_from']}")
                    
                    label = "\\n".join(label_parts)
                    
                    f.write(f'  {node_id} [label="{label}", fillcolor="{color}", style="{style}"];\n')
                
                # Write edges
                f.write("\n")
                for parent_id, child_id in graph.edges:
                    edge_style = "bold" if parent_id in best_path_nodes and child_id in best_path_nodes else "solid"
                    edge_color = "red" if edge_style == "bold" else "black"
                    f.write(f'  {parent_id} -> {child_id} [style="{edge_style}", color="{edge_color}"];\n')
                
                # Add legend
                f.write("\n  // Legend\n")
                f.write('  subgraph cluster_legend {\n')
                f.write('    label="Legend";\n')
                f.write('    style=filled;\n')
                f.write('    color=lightgrey;\n')
                f.write('    node [shape=box, style=rounded];\n')
                f.write('    legend_high [label="High Score (≥0.8)", fillcolor="#90EE90", style=filled];\n')
                f.write('    legend_good [label="Good Score (≥0.6)", fillcolor="#FFFFE0", style=filled];\n')
                f.write('    legend_medium [label="Medium Score (≥0.4)", fillcolor="#FFD700", style=filled];\n')
                f.write('    legend_low [label="Low Score (≥0.2)", fillcolor="#FFA500", style=filled];\n')
                f.write('    legend_poor [label="Poor Score (<0.2)", fillcolor="#FFB6C1", style=filled];\n')
                if self.pruned_nodes:
                    f.write('    legend_pruned [label="Pruned", style="filled,dashed", fillcolor="white"];\n')
                if self.merged_nodes:
                    f.write('    legend_merged [label="Merged", style="filled,dotted", fillcolor="white"];\n')
                f.write('  }\n')
                
                f.write("}\n")
            
            logger.info(f"[GraphVisualizer] Exported graph to DOT format: {output_path}")
            logger.info(f"[GraphVisualizer] Graph stats: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        except Exception as e:
            logger.error(f"[GraphVisualizer] Failed to export graph to DOT: {e}", exc_info=True)
    
    def _get_node_color(self, score: float, node_id: int, best_path_nodes: Set[int]) -> str:
        """Get color for node based on score and path membership.
        
        Args:
            score: Node score
            node_id: Node ID
            best_path_nodes: Set of node IDs in the best path
            
        Returns:
            Color string for visualization
        """
        # Highlight best path with brighter colors
        if node_id in best_path_nodes:
            if score >= 0.8:
                return "#00FF00"  # Bright green
            elif score >= 0.6:
                return "#FFFF00"  # Bright yellow
            elif score >= 0.4:
                return "#FFD700"  # Gold
            else:
                return "#FFA500"  # Orange
        
        # Regular colors for other nodes
        if score >= 0.8:
            return "#90EE90"  # Light green
        elif score >= 0.6:
            return "#FFFFE0"  # Light yellow
        elif score >= 0.4:
            return "#FFD700"  # Gold
        elif score >= 0.2:
            return "#FFA500"  # Orange
        else:
            return "#FFB6C1"  # Light red
    
    def _get_node_style(self, node_id: int) -> str:
        """Get style for node based on its status.
        
        Args:
            node_id: Node ID
            
        Returns:
            Style string for visualization
        """
        if node_id in self.pruned_nodes:
            return "filled,dashed"
        elif node_id in self.merged_nodes:
            return "filled,dotted"
        else:
            return "filled"
    
    def export_to_html(
        self,
        graph: Any,
        output_path: str,
        include_stats: bool = True
    ) -> None:
        """Export graph to interactive HTML visualization.
        
        Args:
            graph: ReasoningGraph object to visualize
            output_path: Path to save the HTML file
            include_stats: Whether to include statistics panel
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare graph data for visualization
            nodes_data = []
            for node_id, node in graph.nodes.items():
                nodes_data.append({
                    'id': node_id,
                    'label': f"Node {node_id}",
                    'score': node.score,
                    'color': self._get_html_color(node.score),
                    'metadata': node.metadata,
                    'is_pruned': node_id in self.pruned_nodes,
                    'is_merged': node_id in self.merged_nodes,
                })
            
            edges_data = []
            for parent_id, child_id in graph.edges:
                edges_data.append({
                    'from': parent_id,
                    'to': child_id,
                })
            
            stats = graph.get_statistics() if include_stats else {}
            
            # Generate HTML
            html_content = self._generate_html_template(nodes_data, edges_data, stats)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"[GraphVisualizer] Exported graph to HTML: {output_path}")
        
        except Exception as e:
            logger.error(f"[GraphVisualizer] Failed to export graph to HTML: {e}", exc_info=True)
    
    def _get_html_color(self, score: float) -> str:
        """Get color for HTML visualization.
        
        Args:
            score: Node score
            
        Returns:
            Color string
        """
        if score >= 0.8:
            return "#90EE90"
        elif score >= 0.6:
            return "#FFFFE0"
        elif score >= 0.4:
            return "#FFD700"
        elif score >= 0.2:
            return "#FFA500"
        else:
            return "#FFB6C1"
    
    def _generate_html_template(
        self,
        nodes_data: List[Dict],
        edges_data: List[Dict],
        stats: Dict[str, Any]
    ) -> str:
        """Generate HTML template for interactive visualization.
        
        Args:
            nodes_data: List of node data dictionaries
            edges_data: List of edge data dictionaries
            stats: Graph statistics
            
        Returns:
            HTML content as string
        """
        nodes_json = json.dumps(nodes_data, indent=2)
        edges_json = json.dumps(edges_data, indent=2)
        stats_json = json.dumps(stats, indent=2)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Reasoning Graph Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        #header {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        #container {{
            display: flex;
            gap: 20px;
        }}
        #graph {{
            flex: 1;
            height: 600px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        #sidebar {{
            width: 300px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-y: auto;
            max-height: 600px;
        }}
        #sidebar h2 {{
            margin-top: 0;
            color: #333;
            font-size: 18px;
        }}
        .stat-item {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .stat-label {{
            font-weight: bold;
            color: #666;
        }}
        .stat-value {{
            color: #333;
            font-size: 18px;
        }}
        #node-info {{
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4f8;
            border-radius: 4px;
            display: none;
        }}
        #node-info h3 {{
            margin-top: 0;
            color: #333;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Reasoning Graph Visualization</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div id="container">
        <div id="graph"></div>
        <div id="sidebar">
            <h2>Graph Statistics</h2>
            <div class="stat-item">
                <div class="stat-label">Total Nodes</div>
                <div class="stat-value">{stats.get('num_nodes', 0)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Edges</div>
                <div class="stat-value">{stats.get('num_edges', 0)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Leaf Nodes</div>
                <div class="stat-value">{stats.get('num_leaves', 0)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Paths</div>
                <div class="stat-value">{stats.get('num_paths', 0)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Max Depth</div>
                <div class="stat-value">{stats.get('max_depth', 0)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Average Score</div>
                <div class="stat-value">{stats.get('avg_score', 0.0):.3f}</div>
            </div>
            
            <div class="legend">
                <h3>Legend</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #90EE90;"></div>
                    <span>High Score (≥0.8)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFFFE0;"></div>
                    <span>Good Score (≥0.6)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFD700;"></div>
                    <span>Medium Score (≥0.4)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFA500;"></div>
                    <span>Low Score (≥0.2)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFB6C1;"></div>
                    <span>Poor Score (<0.2)</span>
                </div>
            </div>
            
            <div id="node-info">
                <h3>Node Information</h3>
                <div id="node-details"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Graph data
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        
        // Create nodes for vis.js
        const nodes = new vis.DataSet(nodesData.map(node => ({{
            id: node.id,
            label: `Node ${{node.id}}\\nScore: ${{node.score.toFixed(3)}}`,
            color: {{
                background: node.color,
                border: node.is_pruned ? '#FF0000' : (node.is_merged ? '#0000FF' : '#000000'),
                highlight: {{
                    background: node.color,
                    border: '#FF0000'
                }}
            }},
            borderWidth: node.is_pruned || node.is_merged ? 3 : 1,
            borderWidthSelected: 4,
            font: {{
                size: 14,
                color: '#333'
            }},
            title: `Node ${{node.id}}<br>Score: ${{node.score.toFixed(3)}}<br>${{JSON.stringify(node.metadata)}}`,
            metadata: node.metadata,
            score: node.score
        }})));
        
        // Create edges for vis.js
        const edges = new vis.DataSet(edgesData.map(edge => ({{
            from: edge.from,
            to: edge.to,
            arrows: 'to',
            color: {{
                color: '#848484',
                highlight: '#FF0000'
            }}
        }})));
        
        // Create network
        const container = document.getElementById('graph');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    nodeSpacing: 150,
                    levelSeparation: 150
                }}
            }},
            physics: {{
                enabled: false
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        
        // Handle node selection
        network.on('selectNode', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const nodeData = nodesData.find(n => n.id === nodeId);
                
                const nodeInfo = document.getElementById('node-info');
                const nodeDetails = document.getElementById('node-details');
                
                let detailsHtml = `
                    <p><strong>Node ID:</strong> ${{nodeData.id}}</p>
                    <p><strong>Score:</strong> ${{nodeData.score.toFixed(4)}}</p>
                    <p><strong>Pruned:</strong> ${{nodeData.is_pruned ? 'Yes' : 'No'}}</p>
                    <p><strong>Merged:</strong> ${{nodeData.is_merged ? 'Yes' : 'No'}}</p>
                `;
                
                if (Object.keys(nodeData.metadata).length > 0) {{
                    detailsHtml += '<p><strong>Metadata:</strong></p><pre>' + 
                                   JSON.stringify(nodeData.metadata, null, 2) + '</pre>';
                }}
                
                nodeDetails.innerHTML = detailsHtml;
                nodeInfo.style.display = 'block';
            }}
        }});
        
        // Handle deselection
        network.on('deselectNode', function() {{
            document.getElementById('node-info').style.display = 'none';
        }});
    </script>
</body>
</html>"""
        
        return html
    
    def export_path_genealogy(
        self,
        graph: Any,
        output_path: str,
        format: str = 'json'
    ) -> None:
        """Export path genealogy information.
        
        Args:
            graph: ReasoningGraph object
            output_path: Path to save the output file
            format: Output format ('json' or 'txt')
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Build genealogy data
            genealogy = []
            for leaf_id in graph.leaf_ids:
                path = graph.get_path(leaf_id)
                path_info = {
                    'leaf_id': leaf_id,
                    'path': path,
                    'length': len(path),
                    'score': graph.get_node(leaf_id).score if graph.get_node(leaf_id) else 0.0,
                    'nodes': []
                }
                
                for node_id in path:
                    node = graph.get_node(node_id)
                    if node:
                        path_info['nodes'].append({
                            'node_id': node_id,
                            'score': node.score,
                            'metadata': node.metadata
                        })
                
                genealogy.append(path_info)
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(genealogy, f, indent=2)
            elif format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("Path Genealogy Report\n")
                    f.write("=" * 80 + "\n\n")
                    for i, path_info in enumerate(genealogy, 1):
                        f.write(f"Path {i} (Leaf Node {path_info['leaf_id']}):\n")
                        f.write(f"  Length: {path_info['length']}\n")
                        f.write(f"  Final Score: {path_info['score']:.4f}\n")
                        f.write(f"  Node Sequence: {' -> '.join(map(str, path_info['path']))}\n")
                        f.write("\n")
            
            logger.info(f"[GraphVisualizer] Exported path genealogy to {output_path}")
        
        except Exception as e:
            logger.error(f"[GraphVisualizer] Failed to export path genealogy: {e}", exc_info=True)
    
    def clear_markers(self) -> None:
        """Clear all pruned and merged node markers."""
        self.pruned_nodes.clear()
        self.merged_nodes.clear()
        logger.debug("[GraphVisualizer] Cleared all node markers")

