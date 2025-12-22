"""Path analysis tools for multi-path latent reasoning.

This module provides tools for analyzing path behavior, including score distributions,
diversity metrics, pruning/merging statistics, and path comparisons.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import numpy as np

# Logger setup
logger = logging.getLogger(__name__)


class PathAnalyzer:
    """Analyzer for reasoning path behavior.
    
    This class provides methods to analyze and compare reasoning paths,
    including statistical analysis, diversity metrics, and performance comparisons.
    """
    
    def __init__(self):
        """Initialize the path analyzer."""
        self.path_history: List[Dict[str, Any]] = []
        self.pruning_events: List[Dict[str, Any]] = []
        self.merging_events: List[Dict[str, Any]] = []
        self.score_history: Dict[int, List[float]] = defaultdict(list)
        logger.info("[PathAnalyzer] Initialized path analyzer")
    
    def record_paths(
        self,
        paths: List[Any],
        step: int,
        agent_name: Optional[str] = None
    ) -> None:
        """Record current state of paths for analysis.
        
        Args:
            paths: List of PathState objects
            step: Current step number
            agent_name: Name of current agent (optional)
        """
        path_snapshot = {
            'step': step,
            'agent_name': agent_name,
            'timestamp': datetime.now().isoformat(),
            'num_paths': len(paths),
            'paths': []
        }
        
        for path in paths:
            path_info = {
                'path_id': path.path_id,
                'score': path.score,
                'length': path.get_length(),
                'metadata': path.metadata.copy() if hasattr(path, 'metadata') else {}
            }
            path_snapshot['paths'].append(path_info)
            
            # Record score history
            self.score_history[path.path_id].append(path.score)
        
        self.path_history.append(path_snapshot)
        logger.debug(f"[PathAnalyzer] Recorded {len(paths)} paths at step {step}")
    
    def record_pruning(
        self,
        pruned_path_ids: List[int],
        remaining_path_ids: List[int],
        step: int,
        reason: Optional[str] = None
    ) -> None:
        """Record a pruning event.
        
        Args:
            pruned_path_ids: List of path IDs that were pruned
            remaining_path_ids: List of path IDs that remain
            step: Current step number
            reason: Reason for pruning (optional)
        """
        event = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'pruned_paths': pruned_path_ids,
            'remaining_paths': remaining_path_ids,
            'num_pruned': len(pruned_path_ids),
            'num_remaining': len(remaining_path_ids),
            'reason': reason
        }
        self.pruning_events.append(event)
        logger.info(f"[PathAnalyzer] Recorded pruning event: {len(pruned_path_ids)} paths pruned at step {step}")
    
    def record_merging(
        self,
        merged_path_ids: List[int],
        new_path_id: int,
        step: int,
        strategy: Optional[str] = None
    ) -> None:
        """Record a merging event.
        
        Args:
            merged_path_ids: List of path IDs that were merged
            new_path_id: ID of the newly created merged path
            step: Current step number
            strategy: Merging strategy used (optional)
        """
        event = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'merged_paths': merged_path_ids,
            'new_path': new_path_id,
            'num_merged': len(merged_path_ids),
            'strategy': strategy
        }
        self.merging_events.append(event)
        logger.info(f"[PathAnalyzer] Recorded merging event: {len(merged_path_ids)} paths merged at step {step}")
    
    def compute_score_statistics(self) -> Dict[str, Any]:
        """Compute statistics about path scores.
        
        Returns:
            Dictionary containing score statistics
        """
        if not self.path_history:
            logger.warning("[PathAnalyzer] No path history to analyze")
            return {}
        
        all_scores = []
        for snapshot in self.path_history:
            for path in snapshot['paths']:
                all_scores.append(path['score'])
        
        if not all_scores:
            return {}
        
        stats = {
            'mean': float(np.mean(all_scores)),
            'std': float(np.std(all_scores)),
            'min': float(np.min(all_scores)),
            'max': float(np.max(all_scores)),
            'median': float(np.median(all_scores)),
            'q25': float(np.percentile(all_scores, 25)),
            'q75': float(np.percentile(all_scores, 75)),
        }
        
        logger.debug(f"[PathAnalyzer] Score statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        return stats
    
    def compute_diversity_metrics(self, step: Optional[int] = None) -> Dict[str, float]:
        """Compute diversity metrics for paths at a given step.
        
        Args:
            step: Step number to analyze (None for latest)
            
        Returns:
            Dictionary containing diversity metrics
        """
        if not self.path_history:
            logger.warning("[PathAnalyzer] No path history to analyze")
            return {}
        
        # Get snapshot
        if step is None:
            snapshot = self.path_history[-1]
        else:
            snapshots = [s for s in self.path_history if s['step'] == step]
            if not snapshots:
                logger.warning(f"[PathAnalyzer] No snapshot found for step {step}")
                return {}
            snapshot = snapshots[0]
        
        scores = [path['score'] for path in snapshot['paths']]
        
        if len(scores) < 2:
            return {'score_variance': 0.0, 'score_range': 0.0}
        
        metrics = {
            'score_variance': float(np.var(scores)),
            'score_range': float(np.max(scores) - np.min(scores)),
            'score_std': float(np.std(scores)),
            'num_paths': len(scores)
        }
        
        logger.debug(f"[PathAnalyzer] Diversity metrics at step {snapshot['step']}: {metrics}")
        return metrics
    
    def analyze_pruning_statistics(self) -> Dict[str, Any]:
        """Analyze pruning events and compute statistics.
        
        Returns:
            Dictionary containing pruning statistics
        """
        if not self.pruning_events:
            logger.info("[PathAnalyzer] No pruning events to analyze")
            return {
                'total_events': 0,
                'total_pruned': 0,
                'avg_pruned_per_event': 0.0,
                'pruning_rate': 0.0
            }
        
        total_pruned = sum(event['num_pruned'] for event in self.pruning_events)
        avg_pruned = total_pruned / len(self.pruning_events)
        
        # Calculate pruning rate (paths pruned / paths before pruning)
        pruning_rates = []
        for event in self.pruning_events:
            total_before = event['num_pruned'] + event['num_remaining']
            if total_before > 0:
                rate = event['num_pruned'] / total_before
                pruning_rates.append(rate)
        
        avg_rate = np.mean(pruning_rates) if pruning_rates else 0.0
        
        stats = {
            'total_events': len(self.pruning_events),
            'total_pruned': total_pruned,
            'avg_pruned_per_event': avg_pruned,
            'pruning_rate': float(avg_rate),
            'events': self.pruning_events
        }
        
        logger.info(f"[PathAnalyzer] Pruning statistics: {len(self.pruning_events)} events, "
                   f"{total_pruned} total pruned, {avg_rate:.2%} avg rate")
        return stats
    
    def analyze_merging_statistics(self) -> Dict[str, Any]:
        """Analyze merging events and compute statistics.
        
        Returns:
            Dictionary containing merging statistics
        """
        if not self.merging_events:
            logger.info("[PathAnalyzer] No merging events to analyze")
            return {
                'total_events': 0,
                'total_merged': 0,
                'avg_merged_per_event': 0.0
            }
        
        total_merged = sum(event['num_merged'] for event in self.merging_events)
        avg_merged = total_merged / len(self.merging_events)
        
        # Count strategies used
        strategies = defaultdict(int)
        for event in self.merging_events:
            if event['strategy']:
                strategies[event['strategy']] += 1
        
        stats = {
            'total_events': len(self.merging_events),
            'total_merged': total_merged,
            'avg_merged_per_event': avg_merged,
            'strategies_used': dict(strategies),
            'events': self.merging_events
        }
        
        logger.info(f"[PathAnalyzer] Merging statistics: {len(self.merging_events)} events, "
                   f"{total_merged} total merged")
        return stats
    
    def compare_paths(
        self,
        correct_path_ids: List[int],
        incorrect_path_ids: List[int]
    ) -> Dict[str, Any]:
        """Compare paths that led to correct vs incorrect answers.
        
        Args:
            correct_path_ids: List of path IDs that led to correct answers
            incorrect_path_ids: List of path IDs that led to incorrect answers
            
        Returns:
            Dictionary containing comparison results
        """
        correct_scores = []
        incorrect_scores = []
        
        # Collect final scores for each path
        for path_id in correct_path_ids:
            if path_id in self.score_history and self.score_history[path_id]:
                correct_scores.append(self.score_history[path_id][-1])
        
        for path_id in incorrect_path_ids:
            if path_id in self.score_history and self.score_history[path_id]:
                incorrect_scores.append(self.score_history[path_id][-1])
        
        comparison = {
            'num_correct': len(correct_path_ids),
            'num_incorrect': len(incorrect_path_ids),
            'correct_scores': {
                'mean': float(np.mean(correct_scores)) if correct_scores else 0.0,
                'std': float(np.std(correct_scores)) if correct_scores else 0.0,
                'min': float(np.min(correct_scores)) if correct_scores else 0.0,
                'max': float(np.max(correct_scores)) if correct_scores else 0.0,
            },
            'incorrect_scores': {
                'mean': float(np.mean(incorrect_scores)) if incorrect_scores else 0.0,
                'std': float(np.std(incorrect_scores)) if incorrect_scores else 0.0,
                'min': float(np.min(incorrect_scores)) if incorrect_scores else 0.0,
                'max': float(np.max(incorrect_scores)) if incorrect_scores else 0.0,
            }
        }
        
        # Calculate score difference
        if correct_scores and incorrect_scores:
            comparison['score_difference'] = comparison['correct_scores']['mean'] - comparison['incorrect_scores']['mean']
        else:
            comparison['score_difference'] = 0.0
        
        logger.info(f"[PathAnalyzer] Path comparison: correct mean={comparison['correct_scores']['mean']:.4f}, "
                   f"incorrect mean={comparison['incorrect_scores']['mean']:.4f}")
        return comparison
    
    def generate_analysis_report(self, output_path: str) -> None:
        """Generate comprehensive analysis report.
        
        Args:
            output_path: Path to save the report
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_snapshots': len(self.path_history),
                    'total_pruning_events': len(self.pruning_events),
                    'total_merging_events': len(self.merging_events),
                },
                'score_statistics': self.compute_score_statistics(),
                'pruning_statistics': self.analyze_pruning_statistics(),
                'merging_statistics': self.analyze_merging_statistics(),
                'path_history': self.path_history,
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"[PathAnalyzer] Generated analysis report: {output_path}")
        
        except Exception as e:
            logger.error(f"[PathAnalyzer] Failed to generate analysis report: {e}", exc_info=True)
    
    def generate_text_report(self, output_path: str) -> None:
        """Generate human-readable text report.
        
        Args:
            output_path: Path to save the report
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("Path Analysis Report\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary
                f.write("SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Snapshots: {len(self.path_history)}\n")
                f.write(f"Total Pruning Events: {len(self.pruning_events)}\n")
                f.write(f"Total Merging Events: {len(self.merging_events)}\n\n")
                
                # Score Statistics
                score_stats = self.compute_score_statistics()
                if score_stats:
                    f.write("SCORE STATISTICS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Mean:   {score_stats['mean']:.4f}\n")
                    f.write(f"Std:    {score_stats['std']:.4f}\n")
                    f.write(f"Min:    {score_stats['min']:.4f}\n")
                    f.write(f"Max:    {score_stats['max']:.4f}\n")
                    f.write(f"Median: {score_stats['median']:.4f}\n")
                    f.write(f"Q25:    {score_stats['q25']:.4f}\n")
                    f.write(f"Q75:    {score_stats['q75']:.4f}\n\n")
                
                # Pruning Statistics
                pruning_stats = self.analyze_pruning_statistics()
                f.write("PRUNING STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Events: {pruning_stats['total_events']}\n")
                f.write(f"Total Pruned: {pruning_stats['total_pruned']}\n")
                f.write(f"Average Pruned per Event: {pruning_stats['avg_pruned_per_event']:.2f}\n")
                f.write(f"Average Pruning Rate: {pruning_stats['pruning_rate']:.2%}\n\n")
                
                # Merging Statistics
                merging_stats = self.analyze_merging_statistics()
                f.write("MERGING STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Events: {merging_stats['total_events']}\n")
                f.write(f"Total Merged: {merging_stats['total_merged']}\n")
                f.write(f"Average Merged per Event: {merging_stats['avg_merged_per_event']:.2f}\n")
                if merging_stats.get('strategies_used'):
                    f.write("Strategies Used:\n")
                    for strategy, count in merging_stats['strategies_used'].items():
                        f.write(f"  - {strategy}: {count}\n")
                f.write("\n")
                
                # Path Evolution
                f.write("PATH EVOLUTION\n")
                f.write("-" * 80 + "\n")
                for snapshot in self.path_history:
                    f.write(f"Step {snapshot['step']}")
                    if snapshot['agent_name']:
                        f.write(f" ({snapshot['agent_name']})")
                    f.write(f": {snapshot['num_paths']} paths\n")
                    
                    if snapshot['paths']:
                        scores = [p['score'] for p in snapshot['paths']]
                        f.write(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}], "
                               f"mean: {np.mean(scores):.4f}\n")
                f.write("\n")
                
                f.write("=" * 80 + "\n")
            
            logger.info(f"[PathAnalyzer] Generated text report: {output_path}")
        
        except Exception as e:
            logger.error(f"[PathAnalyzer] Failed to generate text report: {e}", exc_info=True)
    
    def plot_score_distribution(self, output_path: str) -> None:
        """Plot score distribution across paths.
        
        Args:
            output_path: Path to save the plot
        """
        try:
            # Try to import matplotlib
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
            except ImportError:
                logger.warning("[PathAnalyzer] matplotlib not available, skipping plot generation")
                return
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.path_history:
                logger.warning("[PathAnalyzer] No path history to plot")
                return
            
            # Collect all scores
            all_scores = []
            for snapshot in self.path_history:
                for path in snapshot['paths']:
                    all_scores.append(path['score'])
            
            if not all_scores:
                logger.warning("[PathAnalyzer] No scores to plot")
                return
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(all_scores, bins=20, edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Score Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(all_scores, vert=True)
            ax2.set_ylabel('Score')
            ax2.set_title('Score Box Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[PathAnalyzer] Saved score distribution plot: {output_path}")
        
        except Exception as e:
            logger.error(f"[PathAnalyzer] Failed to plot score distribution: {e}", exc_info=True)
    
    def plot_diversity_over_time(self, output_path: str) -> None:
        """Plot diversity metrics over time.
        
        Args:
            output_path: Path to save the plot
        """
        try:
            # Try to import matplotlib
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
            except ImportError:
                logger.warning("[PathAnalyzer] matplotlib not available, skipping plot generation")
                return
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.path_history:
                logger.warning("[PathAnalyzer] No path history to plot")
                return
            
            # Compute diversity for each snapshot
            steps = []
            variances = []
            ranges = []
            num_paths = []
            
            for snapshot in self.path_history:
                scores = [p['score'] for p in snapshot['paths']]
                if len(scores) >= 2:
                    steps.append(snapshot['step'])
                    variances.append(np.var(scores))
                    ranges.append(np.max(scores) - np.min(scores))
                    num_paths.append(len(scores))
            
            if not steps:
                logger.warning("[PathAnalyzer] Not enough data to plot diversity")
                return
            
            # Create plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
            
            # Score variance
            ax1.plot(steps, variances, marker='o', linewidth=2)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Score Variance')
            ax1.set_title('Score Variance Over Time')
            ax1.grid(True, alpha=0.3)
            
            # Score range
            ax2.plot(steps, ranges, marker='s', linewidth=2, color='orange')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Score Range')
            ax2.set_title('Score Range Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Number of paths
            ax3.plot(steps, num_paths, marker='^', linewidth=2, color='green')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Number of Paths')
            ax3.set_title('Active Paths Over Time')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[PathAnalyzer] Saved diversity over time plot: {output_path}")
        
        except Exception as e:
            logger.error(f"[PathAnalyzer] Failed to plot diversity over time: {e}", exc_info=True)
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self.path_history.clear()
        self.pruning_events.clear()
        self.merging_events.clear()
        self.score_history.clear()
        logger.info("[PathAnalyzer] Cleared all analysis data")

