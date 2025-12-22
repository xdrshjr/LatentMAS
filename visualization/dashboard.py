"""Real-time monitoring dashboard for multi-path latent reasoning.

This module provides a dashboard for monitoring active paths, pruning/merging events,
computational budget usage, and per-agent statistics during execution.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import time

# Logger setup
logger = logging.getLogger(__name__)


class ReasoningDashboard:
    """Real-time monitoring dashboard for multi-path reasoning.
    
    This class tracks and displays information about active paths, events,
    computational costs, and per-agent statistics during execution.
    """
    
    def __init__(self, enable_realtime: bool = False):
        """Initialize the reasoning dashboard.
        
        Args:
            enable_realtime: Whether to enable real-time updates
        """
        self.enable_realtime = enable_realtime
        self.start_time = time.time()
        
        # State tracking
        self.active_paths: Dict[int, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'num_calls': 0,
            'total_paths_processed': 0,
            'total_paths_created': 0,
            'total_paths_pruned': 0,
            'total_paths_merged': 0,
            'total_time': 0.0,
        })
        
        # Budget tracking
        self.computational_budget = {
            'total_tokens': 0,
            'total_forward_passes': 0,
            'total_time': 0.0,
            'budget_limit': None,
            'budget_used_ratio': 0.0,
        }
        
        logger.info("[ReasoningDashboard] Initialized dashboard")
    
    def update_active_paths(self, paths: List[Any]) -> None:
        """Update information about currently active paths.
        
        Args:
            paths: List of PathState objects
        """
        self.active_paths.clear()
        
        for path in paths:
            self.active_paths[path.path_id] = {
                'path_id': path.path_id,
                'score': path.score,
                'length': path.get_length() if hasattr(path, 'get_length') else 0,
                'metadata': path.metadata.copy() if hasattr(path, 'metadata') else {},
                'updated_at': datetime.now().isoformat(),
            }
        
        logger.debug(f"[ReasoningDashboard] Updated {len(paths)} active paths")
    
    def log_event(
        self,
        event_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an event to the dashboard.
        
        Args:
            event_type: Type of event ('pruning', 'merging', 'branching', 'scoring', etc.)
            description: Human-readable description
            details: Additional event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            'type': event_type,
            'description': description,
            'details': details or {}
        }
        
        self.events.append(event)
        logger.info(f"[ReasoningDashboard] Event logged: {event_type} - {description}")
    
    def log_pruning_event(
        self,
        num_pruned: int,
        num_remaining: int,
        pruned_ids: List[int],
        agent_name: Optional[str] = None
    ) -> None:
        """Log a pruning event.
        
        Args:
            num_pruned: Number of paths pruned
            num_remaining: Number of paths remaining
            pruned_ids: List of pruned path IDs
            agent_name: Name of the agent (optional)
        """
        description = f"Pruned {num_pruned} paths, {num_remaining} remaining"
        if agent_name:
            description = f"[{agent_name}] {description}"
        
        self.log_event('pruning', description, {
            'num_pruned': num_pruned,
            'num_remaining': num_remaining,
            'pruned_ids': pruned_ids,
            'agent_name': agent_name,
        })
        
        # Update agent stats
        if agent_name:
            self.agent_stats[agent_name]['total_paths_pruned'] += num_pruned
    
    def log_merging_event(
        self,
        num_merged: int,
        merged_ids: List[int],
        new_id: int,
        strategy: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> None:
        """Log a merging event.
        
        Args:
            num_merged: Number of paths merged
            merged_ids: List of merged path IDs
            new_id: ID of the new merged path
            strategy: Merging strategy used
            agent_name: Name of the agent (optional)
        """
        description = f"Merged {num_merged} paths into path {new_id}"
        if strategy:
            description += f" using {strategy}"
        if agent_name:
            description = f"[{agent_name}] {description}"
        
        self.log_event('merging', description, {
            'num_merged': num_merged,
            'merged_ids': merged_ids,
            'new_id': new_id,
            'strategy': strategy,
            'agent_name': agent_name,
        })
        
        # Update agent stats
        if agent_name:
            self.agent_stats[agent_name]['total_paths_merged'] += num_merged
    
    def log_branching_event(
        self,
        source_id: int,
        num_branches: int,
        branch_ids: List[int],
        agent_name: Optional[str] = None
    ) -> None:
        """Log a branching event.
        
        Args:
            source_id: Source path ID
            num_branches: Number of branches created
            branch_ids: List of new branch IDs
            agent_name: Name of the agent (optional)
        """
        description = f"Branched path {source_id} into {num_branches} new paths"
        if agent_name:
            description = f"[{agent_name}] {description}"
        
        self.log_event('branching', description, {
            'source_id': source_id,
            'num_branches': num_branches,
            'branch_ids': branch_ids,
            'agent_name': agent_name,
        })
        
        # Update agent stats
        if agent_name:
            self.agent_stats[agent_name]['total_paths_created'] += num_branches
    
    def log_scoring_event(
        self,
        path_id: int,
        score: float,
        scoring_method: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> None:
        """Log a scoring event.
        
        Args:
            path_id: Path ID that was scored
            score: Score value
            scoring_method: Method used for scoring
            agent_name: Name of the agent (optional)
        """
        description = f"Scored path {path_id}: {score:.4f}"
        if scoring_method:
            description += f" ({scoring_method})"
        if agent_name:
            description = f"[{agent_name}] {description}"
        
        self.log_event('scoring', description, {
            'path_id': path_id,
            'score': score,
            'scoring_method': scoring_method,
            'agent_name': agent_name,
        })
    
    def update_agent_stats(
        self,
        agent_name: str,
        num_paths_processed: int,
        processing_time: float
    ) -> None:
        """Update statistics for an agent.
        
        Args:
            agent_name: Name of the agent
            num_paths_processed: Number of paths processed
            processing_time: Time taken in seconds
        """
        stats = self.agent_stats[agent_name]
        stats['num_calls'] += 1
        stats['total_paths_processed'] += num_paths_processed
        stats['total_time'] += processing_time
        
        logger.debug(f"[ReasoningDashboard] Updated stats for agent '{agent_name}': "
                    f"{num_paths_processed} paths, {processing_time:.2f}s")
    
    def update_computational_budget(
        self,
        tokens: int = 0,
        forward_passes: int = 0,
        time_elapsed: float = 0.0,
        budget_limit: Optional[int] = None
    ) -> None:
        """Update computational budget usage.
        
        Args:
            tokens: Number of tokens processed
            forward_passes: Number of forward passes
            time_elapsed: Time elapsed in seconds
            budget_limit: Budget limit (if any)
        """
        self.computational_budget['total_tokens'] += tokens
        self.computational_budget['total_forward_passes'] += forward_passes
        self.computational_budget['total_time'] += time_elapsed
        
        if budget_limit is not None:
            self.computational_budget['budget_limit'] = budget_limit
            self.computational_budget['budget_used_ratio'] = (
                self.computational_budget['total_tokens'] / budget_limit
                if budget_limit > 0 else 0.0
            )
        
        logger.debug(f"[ReasoningDashboard] Budget update: {tokens} tokens, "
                    f"{forward_passes} forward passes, {time_elapsed:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current dashboard summary.
        
        Returns:
            Dictionary containing dashboard summary
        """
        elapsed_time = time.time() - self.start_time
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'active_paths': {
                'count': len(self.active_paths),
                'paths': list(self.active_paths.values()),
            },
            'events': {
                'total': len(self.events),
                'by_type': self._count_events_by_type(),
                'recent': self.events[-10:] if self.events else [],
            },
            'agent_stats': dict(self.agent_stats),
            'computational_budget': self.computational_budget.copy(),
        }
        
        return summary
    
    def _count_events_by_type(self) -> Dict[str, int]:
        """Count events by type.
        
        Returns:
            Dictionary mapping event type to count
        """
        counts = defaultdict(int)
        for event in self.events:
            counts[event['type']] += 1
        return dict(counts)
    
    def export_to_json(self, output_path: str) -> None:
        """Export dashboard data to JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            summary = self.get_summary()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"[ReasoningDashboard] Exported dashboard to JSON: {output_path}")
        
        except Exception as e:
            logger.error(f"[ReasoningDashboard] Failed to export to JSON: {e}", exc_info=True)
    
    def export_to_html(self, output_path: str) -> None:
        """Export dashboard to interactive HTML file.
        
        Args:
            output_path: Path to save the HTML file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            summary = self.get_summary()
            html_content = self._generate_html_dashboard(summary)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"[ReasoningDashboard] Exported dashboard to HTML: {output_path}")
        
        except Exception as e:
            logger.error(f"[ReasoningDashboard] Failed to export to HTML: {e}", exc_info=True)
    
    def _generate_html_dashboard(self, summary: Dict[str, Any]) -> str:
        """Generate HTML dashboard.
        
        Args:
            summary: Dashboard summary data
            
        Returns:
            HTML content as string
        """
        # Format elapsed time
        elapsed = summary['elapsed_time']
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        
        # Active paths table
        paths_html = ""
        if summary['active_paths']['paths']:
            paths_html = "<table>"
            paths_html += "<tr><th>Path ID</th><th>Score</th><th>Length</th><th>Updated</th></tr>"
            for path in sorted(summary['active_paths']['paths'], key=lambda p: p['score'], reverse=True):
                paths_html += f"<tr><td>{path['path_id']}</td><td>{path['score']:.4f}</td>"
                paths_html += f"<td>{path['length']}</td><td>{path['updated_at']}</td></tr>"
            paths_html += "</table>"
        else:
            paths_html = "<p>No active paths</p>"
        
        # Agent stats table
        agent_stats_html = ""
        if summary['agent_stats']:
            agent_stats_html = "<table>"
            agent_stats_html += "<tr><th>Agent</th><th>Calls</th><th>Processed</th><th>Created</th><th>Pruned</th><th>Merged</th><th>Time (s)</th></tr>"
            for agent_name, stats in summary['agent_stats'].items():
                agent_stats_html += f"<tr><td>{agent_name}</td><td>{stats['num_calls']}</td>"
                agent_stats_html += f"<td>{stats['total_paths_processed']}</td>"
                agent_stats_html += f"<td>{stats['total_paths_created']}</td>"
                agent_stats_html += f"<td>{stats['total_paths_pruned']}</td>"
                agent_stats_html += f"<td>{stats['total_paths_merged']}</td>"
                agent_stats_html += f"<td>{stats['total_time']:.2f}</td></tr>"
            agent_stats_html += "</table>"
        else:
            agent_stats_html = "<p>No agent statistics</p>"
        
        # Recent events
        events_html = ""
        if summary['events']['recent']:
            events_html = "<ul>"
            for event in reversed(summary['events']['recent']):
                events_html += f"<li><strong>[{event['type']}]</strong> {event['description']} "
                events_html += f"<span class='timestamp'>({event['elapsed_time']:.1f}s)</span></li>"
            events_html += "</ul>"
        else:
            events_html = "<p>No events</p>"
        
        # Budget usage
        budget = summary['computational_budget']
        budget_html = f"""
        <div class="stat-item">
            <div class="stat-label">Total Tokens</div>
            <div class="stat-value">{budget['total_tokens']:,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Forward Passes</div>
            <div class="stat-value">{budget['total_forward_passes']:,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Total Time</div>
            <div class="stat-value">{budget['total_time']:.2f}s</div>
        </div>
        """
        
        if budget['budget_limit']:
            budget_html += f"""
            <div class="stat-item">
                <div class="stat-label">Budget Limit</div>
                <div class="stat-value">{budget['budget_limit']:,}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Budget Used</div>
                <div class="stat-value">{budget['budget_used_ratio']:.1%}</div>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Reasoning Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
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
        .container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .panel {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .panel h2 {{
            margin-top: 0;
            color: #333;
            font-size: 18px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
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
            font-size: 14px;
        }}
        .stat-value {{
            color: #333;
            font-size: 24px;
            margin-top: 5px;
        }}
        ul {{
            list-style-type: none;
            padding: 0;
        }}
        li {{
            padding: 8px;
            margin: 5px 0;
            background-color: #f9f9f9;
            border-radius: 4px;
            border-left: 3px solid #007bff;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
        }}
        .summary-stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .summary-stat {{
            text-align: center;
        }}
        .summary-stat-value {{
            font-size: 36px;
            font-weight: bold;
            color: #007bff;
        }}
        .summary-stat-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Path Reasoning Dashboard</h1>
        <p>Generated: {summary['timestamp']}</p>
        <p>Elapsed Time: {elapsed_str}</p>
        
        <div class="summary-stats">
            <div class="summary-stat">
                <div class="summary-stat-value">{summary['active_paths']['count']}</div>
                <div class="summary-stat-label">Active Paths</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">{summary['events']['total']}</div>
                <div class="summary-stat-label">Total Events</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">{len(summary['agent_stats'])}</div>
                <div class="summary-stat-label">Agents</div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="panel">
            <h2>Active Paths</h2>
            {paths_html}
        </div>
        
        <div class="panel">
            <h2>Computational Budget</h2>
            {budget_html}
        </div>
        
        <div class="panel">
            <h2>Agent Statistics</h2>
            {agent_stats_html}
        </div>
        
        <div class="panel">
            <h2>Recent Events</h2>
            {events_html}
        </div>
        
        <div class="panel">
            <h2>Event Summary</h2>
            <table>
                <tr><th>Event Type</th><th>Count</th></tr>
"""
        
        for event_type, count in summary['events']['by_type'].items():
            html += f"<tr><td>{event_type}</td><td>{count}</td></tr>"
        
        html += """
            </table>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def print_summary(self) -> None:
        """Print dashboard summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("REASONING DASHBOARD SUMMARY")
        print("=" * 80)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Elapsed Time: {summary['elapsed_time']:.2f}s")
        print()
        
        print(f"Active Paths: {summary['active_paths']['count']}")
        if summary['active_paths']['paths']:
            print("  Top 5 paths by score:")
            for path in sorted(summary['active_paths']['paths'], 
                             key=lambda p: p['score'], reverse=True)[:5]:
                print(f"    Path {path['path_id']}: score={path['score']:.4f}, length={path['length']}")
        print()
        
        print(f"Total Events: {summary['events']['total']}")
        print("  By type:")
        for event_type, count in summary['events']['by_type'].items():
            print(f"    {event_type}: {count}")
        print()
        
        if summary['agent_stats']:
            print("Agent Statistics:")
            for agent_name, stats in summary['agent_stats'].items():
                print(f"  {agent_name}:")
                print(f"    Calls: {stats['num_calls']}")
                print(f"    Processed: {stats['total_paths_processed']}")
                print(f"    Created: {stats['total_paths_created']}")
                print(f"    Pruned: {stats['total_paths_pruned']}")
                print(f"    Merged: {stats['total_paths_merged']}")
                print(f"    Time: {stats['total_time']:.2f}s")
        print()
        
        budget = summary['computational_budget']
        print("Computational Budget:")
        print(f"  Total Tokens: {budget['total_tokens']:,}")
        print(f"  Forward Passes: {budget['total_forward_passes']:,}")
        print(f"  Total Time: {budget['total_time']:.2f}s")
        if budget['budget_limit']:
            print(f"  Budget Limit: {budget['budget_limit']:,}")
            print(f"  Budget Used: {budget['budget_used_ratio']:.1%}")
        
        print("=" * 80 + "\n")
    
    def reset(self) -> None:
        """Reset dashboard to initial state."""
        self.start_time = time.time()
        self.active_paths.clear()
        self.events.clear()
        self.agent_stats.clear()
        self.computational_budget = {
            'total_tokens': 0,
            'total_forward_passes': 0,
            'total_time': 0.0,
            'budget_limit': None,
            'budget_used_ratio': 0.0,
        }
        logger.info("[ReasoningDashboard] Reset dashboard")

