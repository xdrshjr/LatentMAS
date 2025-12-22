"""Demonstration of visualization tools for multi-path latent reasoning.

This script demonstrates how to use the visualization tools to analyze
and visualize reasoning graphs and paths.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.graph_structure import ReasoningGraph
from methods.path_manager import PathManager, PathState
from visualization import GraphVisualizer, PathAnalyzer, ReasoningDashboard
import torch


def create_sample_graph():
    """Create a sample reasoning graph for demonstration."""
    graph = ReasoningGraph()
    
    # Create root node
    root_id = graph.add_node(
        hidden_states=torch.randn(1, 10),
        score=0.5,
        metadata={'agent_name': 'root', 'step': 0}
    )
    
    # Create first level (3 branches)
    level1_ids = []
    for i in range(3):
        node_id = graph.add_node(
            hidden_states=torch.randn(1, 10),
            parent_id=root_id,
            score=0.6 + i * 0.1,
            metadata={'agent_name': 'agent_1', 'step': 1, 'branch': i}
        )
        level1_ids.append(node_id)
    
    # Create second level (2 branches from each level1 node)
    level2_ids = []
    for parent_id in level1_ids:
        for i in range(2):
            node_id = graph.add_node(
                hidden_states=torch.randn(1, 10),
                parent_id=parent_id,
                score=0.7 + i * 0.05,
                metadata={'agent_name': 'agent_2', 'step': 2, 'branch': i}
            )
            level2_ids.append(node_id)
    
    # Create final level (1 node from each level2 node)
    for parent_id in level2_ids:
        graph.add_node(
            hidden_states=torch.randn(1, 10),
            parent_id=parent_id,
            score=0.8 + torch.rand(1).item() * 0.2,
            metadata={'agent_name': 'judger', 'step': 3}
        )
    
    return graph


def create_sample_paths():
    """Create sample paths for demonstration."""
    manager = PathManager()
    
    # Create initial paths
    path_ids = []
    for i in range(5):
        path_id = manager.create_path(
            hidden_states=torch.randn(1, 10),
            score=0.5 + i * 0.1,
            metadata={'iteration': 0}
        )
        path_ids.append(path_id)
    
    # Simulate path evolution
    for iteration in range(1, 4):
        for path_id in list(path_ids):
            path = manager.get_path(path_id)
            if path:
                # Update score
                new_score = min(1.0, path.score + torch.rand(1).item() * 0.1)
                path.update_state(score=new_score)
                path.metadata['iteration'] = iteration
    
    return manager


def demo_graph_visualization():
    """Demonstrate graph visualization."""
    print("\n" + "=" * 80)
    print("GRAPH VISUALIZATION DEMO")
    print("=" * 80)
    
    # Create sample graph
    print("\n1. Creating sample reasoning graph...")
    graph = create_sample_graph()
    print(f"   Created graph: {graph}")
    
    # Create visualizer
    visualizer = GraphVisualizer()
    
    # Mark some nodes as pruned/merged for demonstration
    visualizer.mark_pruned([4, 5])
    visualizer.mark_merged([6, 7])
    
    # Export to DOT format
    print("\n2. Exporting to DOT format...")
    output_dir = Path("output/visualization_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dot_path = output_dir / "graph.dot"
    visualizer.export_to_dot(
        graph=graph,
        output_path=str(dot_path),
        show_scores=True,
        show_metadata=True,
        highlight_best_path=True
    )
    print(f"   Saved DOT file: {dot_path}")
    print(f"   To render: dot -Tpng {dot_path} -o graph.png")
    
    # Export to HTML
    print("\n3. Exporting to interactive HTML...")
    html_path = output_dir / "graph.html"
    visualizer.export_to_html(
        graph=graph,
        output_path=str(html_path),
        include_stats=True
    )
    print(f"   Saved HTML file: {html_path}")
    print(f"   Open in browser to interact with the graph")
    
    # Export path genealogy
    print("\n4. Exporting path genealogy...")
    genealogy_path = output_dir / "genealogy.json"
    visualizer.export_path_genealogy(
        graph=graph,
        output_path=str(genealogy_path),
        format='json'
    )
    print(f"   Saved genealogy: {genealogy_path}")
    
    print("\n✓ Graph visualization demo complete!")


def demo_path_analysis():
    """Demonstrate path analysis."""
    print("\n" + "=" * 80)
    print("PATH ANALYSIS DEMO")
    print("=" * 80)
    
    # Create analyzer
    analyzer = PathAnalyzer()
    
    # Create sample paths
    print("\n1. Creating sample paths...")
    manager = create_sample_paths()
    
    # Record path evolution
    print("\n2. Recording path evolution...")
    for step in range(4):
        active_paths = manager.get_active_paths()
        analyzer.record_paths(active_paths, step=step, agent_name=f"agent_{step}")
        print(f"   Step {step}: {len(active_paths)} paths")
    
    # Record some events
    print("\n3. Recording events...")
    analyzer.record_pruning([0, 1], [2, 3, 4], step=2, reason="low_score")
    analyzer.record_merging([2, 3], 5, step=3, strategy="weighted_average")
    
    # Compute statistics
    print("\n4. Computing statistics...")
    score_stats = analyzer.compute_score_statistics()
    print(f"   Score statistics:")
    print(f"     Mean: {score_stats['mean']:.4f}")
    print(f"     Std:  {score_stats['std']:.4f}")
    print(f"     Range: [{score_stats['min']:.4f}, {score_stats['max']:.4f}]")
    
    diversity = analyzer.compute_diversity_metrics()
    print(f"   Diversity metrics:")
    print(f"     Score variance: {diversity['score_variance']:.4f}")
    print(f"     Score range: {diversity['score_range']:.4f}")
    
    pruning_stats = analyzer.analyze_pruning_statistics()
    print(f"   Pruning statistics:")
    print(f"     Total events: {pruning_stats['total_events']}")
    print(f"     Total pruned: {pruning_stats['total_pruned']}")
    print(f"     Avg pruning rate: {pruning_stats['pruning_rate']:.2%}")
    
    # Generate reports
    print("\n5. Generating reports...")
    output_dir = Path("output/visualization_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_report = output_dir / "analysis.json"
    analyzer.generate_analysis_report(str(json_report))
    print(f"   Saved JSON report: {json_report}")
    
    text_report = output_dir / "analysis.txt"
    analyzer.generate_text_report(str(text_report))
    print(f"   Saved text report: {text_report}")
    
    # Generate plots (if matplotlib available)
    print("\n6. Generating plots...")
    try:
        score_plot = output_dir / "score_distribution.png"
        analyzer.plot_score_distribution(str(score_plot))
        print(f"   Saved score distribution: {score_plot}")
        
        diversity_plot = output_dir / "diversity_over_time.png"
        analyzer.plot_diversity_over_time(str(diversity_plot))
        print(f"   Saved diversity plot: {diversity_plot}")
    except Exception as e:
        print(f"   Skipping plots (matplotlib not available or error: {e})")
    
    print("\n✓ Path analysis demo complete!")


def demo_dashboard():
    """Demonstrate dashboard functionality."""
    print("\n" + "=" * 80)
    print("DASHBOARD DEMO")
    print("=" * 80)
    
    # Create dashboard
    dashboard = ReasoningDashboard(enable_realtime=False)
    
    # Create sample paths
    print("\n1. Creating sample paths...")
    manager = create_sample_paths()
    active_paths = manager.get_active_paths()
    
    # Update dashboard
    print("\n2. Updating dashboard with paths...")
    dashboard.update_active_paths(active_paths)
    print(f"   {len(active_paths)} paths tracked")
    
    # Log events
    print("\n3. Logging events...")
    dashboard.log_pruning_event(2, 3, [0, 1], agent_name="agent_1")
    dashboard.log_merging_event(2, [2, 3], 5, strategy="weighted", agent_name="agent_2")
    dashboard.log_branching_event(4, 2, [6, 7], agent_name="agent_2")
    dashboard.log_scoring_event(5, 0.85, scoring_method="ensemble", agent_name="agent_3")
    print("   Logged 4 events")
    
    # Update agent stats
    print("\n4. Updating agent statistics...")
    dashboard.update_agent_stats("agent_1", num_paths_processed=5, processing_time=1.5)
    dashboard.update_agent_stats("agent_2", num_paths_processed=3, processing_time=2.1)
    dashboard.update_agent_stats("agent_3", num_paths_processed=3, processing_time=1.8)
    
    # Update budget
    print("\n5. Updating computational budget...")
    dashboard.update_computational_budget(
        tokens=5000,
        forward_passes=50,
        time_elapsed=5.4,
        budget_limit=10000
    )
    
    # Print summary
    print("\n6. Dashboard summary:")
    dashboard.print_summary()
    
    # Export
    print("\n7. Exporting dashboard...")
    output_dir = Path("output/visualization_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "dashboard.json"
    dashboard.export_to_json(str(json_path))
    print(f"   Saved JSON: {json_path}")
    
    html_path = output_dir / "dashboard.html"
    dashboard.export_to_html(str(html_path))
    print(f"   Saved HTML: {html_path}")
    print(f"   Open in browser to view interactive dashboard")
    
    print("\n✓ Dashboard demo complete!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("VISUALIZATION TOOLS DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases the visualization and analysis tools")
    print("for multi-path latent reasoning.")
    
    # Run demos
    demo_graph_visualization()
    demo_path_analysis()
    demo_dashboard()
    
    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETE!")
    print("=" * 80)
    print("\nOutput files saved to: output/visualization_demo/")
    print("\nFiles generated:")
    print("  - graph.dot (Graphviz DOT format)")
    print("  - graph.html (Interactive graph visualization)")
    print("  - genealogy.json (Path genealogy data)")
    print("  - analysis.json (Analysis report)")
    print("  - analysis.txt (Human-readable report)")
    print("  - score_distribution.png (Score histogram, if matplotlib available)")
    print("  - diversity_over_time.png (Diversity timeline, if matplotlib available)")
    print("  - dashboard.json (Dashboard data)")
    print("  - dashboard.html (Interactive dashboard)")
    print("\n")


if __name__ == "__main__":
    main()

