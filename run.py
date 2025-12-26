import argparse
import json
import logging
import sys
from typing import Dict, List, Tuple, Optional
import torch

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.latent_mas_multipath import LatentMASMultiPathMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed, create_output_file_path, save_question_answer_record, create_result_log_file_path, save_to_csv_results
from config import ConfigLoader, MultiPathConfig, list_presets, get_preset_description
from logging_config import setup_logging, create_log_file_path
from progress_utils import get_progress_manager, reset_progress_manager
from visualization.graph_viz import GraphVisualizer
import time
from pathlib import Path
from datetime import datetime

# Logger will be configured in main()
logger = logging.getLogger(__name__)


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    """Evaluate predictions and calculate accuracy.
    
    Args:
        preds: List of prediction dictionaries, each containing a 'correct' field.
        
    Returns:
        Tuple of (accuracy, correct_count).
    """
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    logger.debug(f"Evaluation: {correct}/{total} correct, accuracy: {acc:.4f}")
    return acc, correct


def _build_graph_from_paths(method, batch_idx: int):
    """Build reasoning graph from path information.
    
    Args:
        method: The LatentMASMultiPathMethod instance
        batch_idx: Index of the item in the current batch
        
    Returns:
        ReasoningGraph object or None if no paths available
    """
    try:
        from methods.graph_structure import ReasoningGraph
        
        # Get all paths from path manager
        path_manager = method.path_manager
        if not path_manager.paths:
            logger.debug(f"[Visualization] No paths available in path manager")
            return None
        
        # Create a new graph
        graph = ReasoningGraph()
        
        # Get all paths (we'll use all paths since we don't have batch_idx mapping)
        all_paths = list(path_manager.paths.values())
        
        if not all_paths:
            logger.debug(f"[Visualization] No paths found")
            return None
        
        logger.debug(f"[Visualization] Building graph from {len(all_paths)} paths")
        
        # Create nodes for each path
        # Each path becomes a leaf node in the graph
        # We'll organize them by agent if metadata is available
        path_to_node = {}  # Map path_id to node_id
        agent_to_nodes = {}  # Map agent_name to list of node_ids
        
        for path in all_paths:
            # Get agent name from metadata, or use path_id as fallback
            agent_name = path.metadata.get('agent_name', f'agent_{path.path_id % 10}')
            agent_idx = path.metadata.get('agent_idx', path.path_id)
            
            # Add node for this path
            node_id = graph.add_node(
                hidden_states=path.hidden_states,
                kv_cache=path.kv_cache,
                parent_id=None,  # We'll set parent relationships later
                score=path.score,
                metadata={
                    'path_id': path.path_id,
                    'agent_name': agent_name,
                    'agent_idx': agent_idx,
                    'step': path.metadata.get('step', 0),
                    'latent_steps': len(path.latent_history),
                    **{k: v for k, v in path.metadata.items() 
                       if k not in ['agent_name', 'agent_idx', 'step'] and 
                       not isinstance(v, torch.Tensor)}
                }
            )
            path_to_node[path.path_id] = node_id
            if not path.node_ids:
                path.node_ids.append(node_id)
            
            # Track nodes by agent
            if agent_name not in agent_to_nodes:
                agent_to_nodes[agent_name] = []
            agent_to_nodes[agent_name].append(node_id)
            
            logger.debug(f"[Visualization] Added node {node_id} for path {path.path_id} "
                        f"(agent: {agent_name}, score: {path.score:.4f})")
        
        # Build hierarchical structure: earlier agents -> later agents
        # Sort agents by minimum path_id in each group (earlier paths come first)
        def get_min_path_id_for_agent(agent_nodes):
            # Find the minimum path_id for nodes in this agent group
            min_path_id = float('inf')
            for node_id in agent_nodes:
                # Find path that corresponds to this node
                for path in all_paths:
                    if path.path_id in path_to_node and path_to_node[path.path_id] == node_id:
                        min_path_id = min(min_path_id, path.path_id)
                        break
            return min_path_id if min_path_id != float('inf') else 0
        
        sorted_agents = sorted(agent_to_nodes.items(), key=lambda x: get_min_path_id_for_agent(x[1]))
        
        # Create parent-child relationships between agents
        for i in range(len(sorted_agents) - 1):
            current_agent, current_nodes = sorted_agents[i]
            next_agent, next_nodes = sorted_agents[i + 1]
            
            # Connect all nodes from current agent to all nodes from next agent
            # This creates a bipartite structure
            for current_node_id in current_nodes:
                for next_node_id in next_nodes:
                    current_node = graph.get_node(current_node_id)
                    next_node = graph.get_node(next_node_id)
                    if current_node and next_node:
                        # Update parent-child relationship
                        if next_node.parent_id is None:
                            next_node.parent_id = current_node_id
                            current_node.add_child(next_node_id)
                            graph.edges.append((current_node_id, next_node_id))
                            graph.leaf_ids.discard(current_node_id)
                            logger.debug(f"[Visualization] Linked node {next_node_id} (agent: {next_agent}) "
                                       f"to parent {current_node_id} (agent: {current_agent})")
        
        logger.info(f"[Visualization] Built graph with {len(graph.nodes)} nodes, "
                   f"{len(graph.edges)} edges from {len(all_paths)} paths across {len(agent_to_nodes)} agents")
        return graph
        
    except Exception as e:
        logger.error(f"[Visualization] Failed to build graph from paths: {e}", exc_info=True)
        logger.debug(f"[Visualization] Graph building error: {type(e).__name__}: {str(e)}")
        return None


def generate_visualizations(
    method,
    batch_idx: int,
    problem_idx: int,
    question: str,
    args: argparse.Namespace,
    output_dir: Optional[str] = None
) -> None:
    """Generate visualization graphs for latent reasoning.
    
    Args:
        method: The method instance (should be LatentMASMultiPathMethod for graph visualization)
        batch_idx: Index of the item in the current batch
        problem_idx: Global problem index
        question: The question text
        args: Command line arguments
        output_dir: Optional output directory for visualization files
    """
    # Check if visualization is enabled
    enable_viz = getattr(args, 'enable_visualization', True)
    if not enable_viz:
        logger.debug(f"[Visualization] Visualization disabled by configuration for problem #{problem_idx}")
        return
    
    # Only generate visualizations for latent_mas_multipath method
    if not isinstance(method, LatentMASMultiPathMethod):
        logger.debug(f"[Visualization] Skipping visualization for method {type(method).__name__}")
        return
    
    try:
        logger.info("=" * 80)
        logger.info(f"[Visualization] Generating visualizations for problem #{problem_idx}")
        logger.info("=" * 80)
        
        # Create output directory
        if output_dir is None:
            model_short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
            task_name = args.task if hasattr(args, 'task') else "custom"
            output_dir = Path("output") / "visualizations" / f"{task_name}_{args.method}_{model_short}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[Visualization] Output directory: {output_dir}")
        
        # Get reasoning graph from method or build from paths
        reasoning_graph = method.reasoning_graph
        if reasoning_graph is None or len(reasoning_graph.nodes) == 0:
            # Try to build graph from path information
            logger.info(f"[Visualization] Graph is empty, building from path information")
            reasoning_graph = _build_graph_from_paths(method, batch_idx)
            if reasoning_graph is None or len(reasoning_graph.nodes) == 0:
                logger.warning(f"[Visualization] No graph data available for problem #{problem_idx}")
                return
        
        logger.info(f"[Visualization] Graph stats: {len(reasoning_graph.nodes)} nodes, "
                   f"{len(reasoning_graph.edges)} edges")
        
        # Initialize visualizers
        graph_viz = GraphVisualizer()
        
        # Generate graph visualization (DOT format)
        dot_file = output_dir / f"problem_{problem_idx}_graph.dot"
        logger.info(f"[Visualization] Exporting graph to DOT format: {dot_file}")
        graph_viz.export_to_dot(
            graph=reasoning_graph,
            output_path=str(dot_file),
            show_scores=True,
            show_metadata=True,
            highlight_best_path=True
        )
        logger.debug(f"[Visualization] DOT file saved: {dot_file}")
        
        # Generate graph visualization (HTML format)
        html_file = output_dir / f"problem_{problem_idx}_graph.html"
        logger.info(f"[Visualization] Exporting graph to HTML format: {html_file}")
        graph_viz.export_to_html(
            graph=reasoning_graph,
            output_path=str(html_file),
            include_stats=True
        )
        logger.debug(f"[Visualization] HTML file saved: {html_file}")
        
        # Generate path genealogy
        genealogy_file = output_dir / f"problem_{problem_idx}_genealogy.json"
        logger.info(f"[Visualization] Exporting path genealogy: {genealogy_file}")
        graph_viz.export_path_genealogy(
            graph=reasoning_graph,
            output_path=str(genealogy_file),
            format='json'
        )
        logger.debug(f"[Visualization] Genealogy file saved: {genealogy_file}")
        
        # Generate path analysis report
        # Note: PathAnalyzer needs path history, which we need to collect from the method
        # For now, we'll generate basic statistics from the graph
        stats = reasoning_graph.get_statistics()
        logger.debug(f"[Visualization] Graph statistics: {stats}")
        
        # Save graph statistics
        stats_file = output_dir / f"problem_{problem_idx}_stats.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'problem_idx': problem_idx,
                    'question': question[:200],  # Truncate long questions
                    'timestamp': datetime.now().isoformat(),
                    'graph_stats': stats,
                }, f, indent=2)
            logger.debug(f"[Visualization] Statistics file saved: {stats_file}")
        except Exception as e:
            logger.warning(f"[Visualization] Failed to save statistics file: {e}")
            logger.debug(f"[Visualization] Statistics save error: {type(e).__name__}: {str(e)}")
        
        logger.info(f"[Visualization] All visualizations generated successfully for problem #{problem_idx}")
        logger.info(f"[Visualization] Output directory: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"[Visualization] Failed to generate visualizations for problem #{problem_idx}: {e}", exc_info=True)
        logger.debug(f"[Visualization] Error details: {type(e).__name__}: {str(e)}")

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    max_samples: int,
    args: argparse.Namespace,
    progress_mgr=None,
    output_file: Optional[str] = None,
) -> Tuple[int, List[Dict]]:
    """Process a batch of questions using the specified method.
    
    Args:
        method: The method instance (BaselineMethod, TextMASMethod, or LatentMASMethod).
        batch: List of question dictionaries to process.
        processed: Number of questions already processed.
        preds: List to accumulate predictions.
        max_samples: Maximum number of samples to process.
        args: Command line arguments.
        progress_mgr: Progress manager for updating progress bar.
        output_file: Optional path to output file for saving question-answer records.
        
    Returns:
        Tuple of (updated_processed_count, updated_predictions_list).
    """
    remaining = max_samples - processed
    if remaining <= 0:
        logger.debug(f"No remaining samples to process (processed: {processed}, max: {max_samples})")
        return processed, preds
    current_batch = batch[:remaining]
    
    logger.info("=" * 80)
    logger.info(f"Processing batch of {len(current_batch)} questions (total processed: {processed}/{max_samples})")
    logger.info("=" * 80)
    
    # Log questions in the batch
    for batch_idx, item in enumerate(current_batch):
        logger.info(f"Batch item {batch_idx + 1}/{len(current_batch)}: {item.get('question', '')[:200]}...")
    
    try:
        if args.method in ["latent_mas", "latent_mas_multipath"] and args.use_vllm: 
            logger.info(f"Using vLLM backend for {args.method} method")
            results = method.run_batch_vllm(current_batch) 
        else:
            logger.info(f"Using standard batch processing for {args.method} method")
            results = method.run_batch(current_batch)
    except Exception as e:
        logger.error(f"Error processing batch: {e}", exc_info=True)
        raise
    
    if len(results) > remaining:
        results = results[:remaining]
        logger.warning(f"Results truncated to {remaining} items")
    
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        
        # Log problem processing with detailed information
        logger.info("=" * 80)
        logger.info(f"COMPLETED PROBLEM #{problem_idx}")
        logger.info(f"  Question: {res.get('question', '')[:100]}...")
        logger.info(f"  Prediction: {res.get('prediction')}")
        logger.info(f"  Gold Answer: {res.get('gold')}")
        logger.info(f"  Result: {'✓ CORRECT' if res.get('correct') else '✗ INCORRECT'}")
        if 'num_paths_used' in res:
            logger.info(f"  Paths Used: {res.get('num_paths_used')}")
        logger.info("=" * 80)
        
        # Save question-answer record to output file
        if output_file:
            save_question_answer_record(
                output_file=output_file,
                problem_idx=problem_idx,
                question=res.get('question', ''),
                prediction=res.get('prediction'),
                gold=res.get('gold'),
                correct=res.get('correct', False),
                additional_info={
                    'method': args.method,
                    'model': args.model_name,
                }
            )
        
        # Show detailed agent trace information
        agents = res.get("agents", [])
        logger.info(f"Problem #{problem_idx} processed by {len(agents)} agents")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Problem #{problem_idx} Question: {res.get('question', '').strip()}")
            logger.debug(f"Problem #{problem_idx} Gold answer: {res.get('gold', '')}")
            
            for agent_idx, a in enumerate(agents):
                name = a.get("name", "Agent")
                role = a.get("role", "")
                logger.debug(f"  Agent {agent_idx + 1}/{len(agents)}: {name} ({role})")
                
                # Log agent input (prompt)
                agent_input = a.get("input", "")
                if agent_input:
                    logger.debug(f"    Input prompt: {agent_input[:200]}...")
                
                # Log agent processing details
                if "latent_steps" in a:
                    logger.debug(f"    Latent steps: {a.get('latent_steps')}")
                if "num_paths" in a:
                    logger.debug(f"    Number of paths: {a.get('num_paths')}")
                if "path_scores" in a:
                    scores = a.get('path_scores', [])
                    if scores:
                        logger.debug(f"    Path scores: {[f'{s:.4f}' for s in scores]}")
                
                # Log agent output
                agent_output = a.get("output", "").rstrip()
                if agent_output:
                    logger.debug(f"    Output: {agent_output}")
                else:
                    logger.debug(f"    Output: (no text output - latent reasoning only)")
        
        # Always show agent summary at INFO level
        for agent_idx, a in enumerate(agents):
            name = a.get("name", "Agent")
            role = a.get("role", "")
            if "num_paths" in a:
                logger.info(f"  Agent {agent_idx + 1}: {name} ({role}) - {a.get('num_paths')} paths")
            elif "num_paths_aggregated" in a:
                logger.info(f"  Agent {agent_idx + 1}: {name} ({role}) - aggregated {a.get('num_paths_aggregated')} paths")
            else:
                logger.info(f"  Agent {agent_idx + 1}: {name} ({role})")
        
        # Generate visualizations for latent_mas_multipath method
        try:
            generate_visualizations(
                method=method,
                batch_idx=offset,
                problem_idx=problem_idx,
                question=res.get('question', ''),
                args=args,
                output_dir=None  # Will use default directory
            )
        except Exception as e:
            # Log warning but don't fail the entire process
            logger.warning(f"[Visualization] Failed to generate visualizations for problem #{problem_idx}: {e}")
            logger.debug(f"[Visualization] Visualization error details: {type(e).__name__}: {str(e)}", exc_info=True)

    processed += len(results)
    
    # Update progress bar with current accuracy
    if progress_mgr is not None:
        correct = sum(1 for p in preds if p.get("correct", False))
        acc = correct / len(preds) if len(preds) > 0 else 0.0
        progress_mgr.update_main_progress(len(results))
        progress_mgr.set_main_postfix(acc=f"{acc:.2%}", correct=f"{correct}/{len(preds)}")
    else:
        # Multi-GPU worker mode: output progress marker for orchestrator to parse
        correct = sum(1 for p in preds if p.get("correct", False))
        acc = correct / len(preds) if len(preds) > 0 else 0.0
        # Output to stderr with special marker format
        sys.stderr.write(f"[PROGRESS:{processed}/{max_samples}|acc:{acc:.4f}|correct:{correct}/{len(preds)}]\n")
        sys.stderr.flush()
    
    logger.debug(f"Batch processing complete. Total processed: {processed}")
    return processed, preds


def run_custom_questions(
    method,
    custom_questions: List[Dict],
    args: argparse.Namespace,
    progress_mgr=None,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """Run inference on a list of custom questions.
    
    Args:
        method: The method instance to use for inference.
        custom_questions: List of question dictionaries. Each dict should have at least a 'question' field.
                          Optional fields: 'gold', 'solution' for evaluation.
        args: Command line arguments.
        progress_mgr: Progress manager for updating progress bar.
        output_file: Optional path to output file for saving question-answer records.
        
    Returns:
        List of prediction dictionaries.
    """
    logger.info(f"Running custom questions mode with {len(custom_questions)} questions")
    logger.debug(f"Method: {args.method}, Model: {args.model_name}")
    
    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    max_samples = len(custom_questions)
    
    for item in custom_questions:
        if processed >= max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                max_samples,
                args,
                progress_mgr,
                output_file,
            )
            batch = []
            if processed >= max_samples:
                break

    if batch and processed < max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            max_samples=max_samples,
            args=args,
            progress_mgr=progress_mgr,
            output_file=output_file,
        )
    
    logger.info(f"Completed processing {len(preds)} custom questions")
    return preds


def save_run_result_log(
    args: argparse.Namespace,
    total_time: float,
    acc: float,
    correct: int,
    total: int,
    custom_questions: Optional[List[Dict]] = None
) -> None:
    """Save run parameters and final results to a log file.
    
    This function records the complete run summary including all parameters
    and final results (accuracy, success rate, etc.) to a dedicated log file
    for later analysis.
    
    Args:
        args: Command line arguments namespace containing all run parameters
        total_time: Total processing time in seconds
        acc: Final accuracy score
        correct: Number of correct predictions
        total: Total number of samples processed
        custom_questions: Optional list of custom questions (to determine if custom mode)
    """
    try:
        # Create result log file path
        task_name = args.task if custom_questions is None else "custom"
        log_file_path = create_result_log_file_path(task_name, args.method)
        
        # Collect all run parameters
        run_params = {
            # Core parameters
            "method": args.method,
            "model_name": args.model_name,
            "task": task_name,
            "split": args.split if custom_questions is None else "custom",
            "max_samples": args.max_samples,
            "seed": args.seed,
            "device": args.device,
            "prompt": getattr(args, 'prompt', 'sequential'),
            
            # Generation parameters
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "generate_bs": args.generate_bs,
            
            # Method-specific parameters
            "latent_steps": getattr(args, 'latent_steps', None),
            "text_mas_context_length": getattr(args, 'text_mas_context_length', -1),
            "think": getattr(args, 'think', False),
            "latent_space_realign": getattr(args, 'latent_space_realign', False),
            
            # vLLM parameters
            "use_vllm": getattr(args, 'use_vllm', False),
            "enable_prefix_caching": getattr(args, 'enable_prefix_caching', False),
            "use_second_HF_model": getattr(args, 'use_second_HF_model', False),
            "device2": getattr(args, 'device2', None),
            "tensor_parallel_size": getattr(args, 'tensor_parallel_size', 1),
            "gpu_memory_utilization": getattr(args, 'gpu_memory_utilization', 0.9),
            
            # Multi-path specific parameters (if applicable)
            "num_paths": getattr(args, 'num_paths', None),
            "enable_branching": getattr(args, 'enable_branching', None),
            "enable_merging": getattr(args, 'enable_merging', None),
            "pruning_strategy": getattr(args, 'pruning_strategy', None),
            "merge_threshold": getattr(args, 'merge_threshold', None),
            "branch_threshold": getattr(args, 'branch_threshold', None),
            "diversity_strategy": getattr(args, 'diversity_strategy', None),
            "latent_consistency_metric": getattr(args, 'latent_consistency_metric', None),
            
            # Configuration file parameters
            "config": getattr(args, 'config', None),
            "config_preset": getattr(args, 'config_preset', None),
            
            # Visualization
            "enable_visualization": getattr(args, 'enable_visualization', True),
        }
        
        # Remove None values for cleaner output
        run_params = {k: v for k, v in run_params.items() if v is not None}
        
        # Final results
        results = {
            "accuracy": acc,
            "success_rate": acc,  # Same as accuracy
            "correct": correct,
            "total": total,
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / total, 4) if total > 0 else 0.0,
        }
        
        # Write to log file
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RUN SUMMARY LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("RUN PARAMETERS\n")
            f.write("-" * 80 + "\n")
            for key, value in sorted(run_params.items()):
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
            f.write(f"  Success Rate: {results['success_rate']:.4f} ({results['success_rate']*100:.2f}%)\n")
            f.write(f"  Correct: {results['correct']}\n")
            f.write(f"  Total: {results['total']}\n")
            f.write(f"  Total Time: {results['total_time_sec']:.4f} seconds\n")
            f.write(f"  Time per Sample: {results['time_per_sample_sec']:.4f} seconds\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("RESULTS JSON\n")
            f.write("-" * 80 + "\n")
            result_json = json.dumps({
                "parameters": run_params,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }, ensure_ascii=False, indent=2)
            f.write(result_json + "\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Run summary saved to: {log_file_path}")
        logger.debug(f"Result log contains parameters and final results for analysis")
        
    except Exception as e:
        logger.error(f"Failed to save run result log: {e}", exc_info=True)
        logger.debug(f"Result log save error: {type(e).__name__}: {str(e)}")


def main(custom_questions: Optional[List[Dict]] = None, args: Optional[argparse.Namespace] = None):
    """Main function to run evaluation on datasets or custom questions.
    
    Args:
        custom_questions: Optional list of custom question dictionaries. If provided, 
                         will run on these questions instead of loading from dataset.
                         Each dict should have at least a 'question' field.
        args: Optional argparse.Namespace object with arguments. If None, will parse from command line.
    """
    if args is None:
        parser = argparse.ArgumentParser()

        # core args for experiments
        parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas", "latent_mas_multipath"], required=True,
                            help="Which multi-agent method to run: 'baseline', 'text_mas', 'latent_mas', or 'latent_mas_multipath'.")
        parser.add_argument("--model_name", type=str, required=True,
                            help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').")
        parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
        parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                            help="Dataset/task to evaluate. Controls which loader is used.")
        parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")
        parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                            help="Set the logging level.")

        # other args
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--split", type=str, default="test")
        parser.add_argument("--max_new_tokens", type=int, default=4096)
        parser.add_argument("--latent_steps", type=int, default=None, help="Number of latent steps for LatentMAS method (default: uses method/config default)")
        parser.add_argument("--temperature", type=float, default=None, help="Baseline temperature for generation and diversity strategies (default: 0.7)")
        parser.add_argument("--top_p", type=float, default=0.95)
        parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
        parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
        parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
        parser.add_argument("--latent_space_realign", action="store_true")
        parser.add_argument("--seed", type=int, default=42)

        # vLLM support
        parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
        parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
        parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
        parser.add_argument("--device2", type=str, default="cuda:1")
        parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")
        
        # Multi-path specific arguments
        parser.add_argument("--num_paths", type=int, default=5, help="Number of parallel reasoning paths for latent_mas_multipath")
        parser.add_argument("--num_parent_paths", type=int, default=5, help="Number of top-scoring parent paths to use for next agent (default: 5)")
        parser.add_argument("--enable_branching", action="store_true", help="Enable adaptive branching in multi-path reasoning")
        parser.add_argument("--enable_merging", action="store_true", help="Enable path merging in multi-path reasoning")
        parser.add_argument("--pruning_strategy", type=str, choices=["topk", "adaptive", "diversity", "budget"], default="adaptive",
                            help="Pruning strategy for multi-path reasoning")
        parser.add_argument("--topk_k", type=int, default=3, help="Number of paths to keep when using topk pruning strategy (only effective when pruning_strategy=topk)")
        parser.add_argument("--merge_threshold", type=float, default=0.9, help="Similarity threshold for path merging")
        parser.add_argument("--branch_threshold", type=float, default=0.5, help="Uncertainty threshold for adaptive branching")
        parser.add_argument("--diversity_strategy", type=str, choices=["temperature", "noise", "hybrid"], default="hybrid",
                            help="Diversity strategy for generating diverse paths")
        parser.add_argument("--latent_consistency_metric", type=str, 
                            choices=["cosine", "euclidean", "l2", "kl_divergence"], default="cosine",
                            help="Similarity metric for latent consistency scoring (cosine/euclidean/l2/kl_divergence)")
        
        # Configuration file support
        parser.add_argument("--config", type=str, default=None, help="Path to configuration file (JSON or YAML)")
        parser.add_argument("--config_preset", type=str, default=None, 
                            choices=["conservative", "balanced", "aggressive", "fast", "quality"],
                            help="Use a preset configuration (conservative/balanced/aggressive/fast/quality)")
        parser.add_argument("--list_presets", action="store_true", help="List available configuration presets and exit")
        
        # Visualization control
        parser.add_argument("--enable_visualization", action="store_true", default=True,
                            help="Enable visualization graph generation (default: True)")
        parser.add_argument("--disable_visualization", dest="enable_visualization", action="store_false",
                            help="Disable visualization graph generation")
        
        # PRM training data collection
        parser.add_argument("--collect_prm_data", action="store_true",
                            help="Enable PRM training data collection mode")
        parser.add_argument("--prm_output_dir", type=str, default="prm_data",
                            help="Output directory for PRM training data (default: prm_data at project root)")
        parser.add_argument("--prm_disable_pruning", action="store_true",
                            help="Disable path pruning in PRM data collection mode (collect all paths)")
        parser.add_argument("--prm_disable_merging", action="store_true",
                            help="Disable path merging in PRM data collection mode (collect all paths)")
        
        # Multi-GPU support (used internally by run_multi_gpu.py)
        parser.add_argument("--gpu_id", type=int, default=None,
                            help="GPU ID for this process (used in multi-GPU mode)")
        parser.add_argument("--data_start_idx", type=int, default=0,
                            help="Starting index in dataset for this GPU (used in multi-GPU mode)")
        parser.add_argument("--output_suffix", type=str, default=None,
                            help="Suffix for output files (e.g., 'gpu_0' for multi-GPU mode)")

        args = parser.parse_args()
        
        # Handle --list_presets
        if args.list_presets:
            print("Available configuration presets for latent_mas_multipath:\n")
            for preset_name in list_presets():
                print(f"  {preset_name}:")
                print(f"    {get_preset_description(preset_name)}")
                print()
            import sys
            sys.exit(0)
    
    # Setup enhanced logging system
    log_file = None
    if not hasattr(args, 'no_log_file') or not args.no_log_file:
        # Extract model short name for log file
        model_short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
        task_name = args.task if custom_questions is None else "custom"
        log_file = create_log_file_path(task_name, args.method, model_short)
    
    # Setup logging with progress bar support
    setup_logging(
        log_level="DEBUG",  # Log everything to file
        console_level=args.log_level,  # But only show specified level in console
        log_file=log_file,
        use_colors=True,
        progress_bar_mode=True
    )
    
    logger.info(f"Logging configured: console_level={args.log_level}, log_file={log_file}")
    
    # Log visualization setting
    enable_viz = getattr(args, 'enable_visualization', True)
    if enable_viz:
        logger.info("[Visualization] Visualization generation is ENABLED")
    else:
        logger.info("[Visualization] Visualization generation is DISABLED")
    
    # Create output file for question-answer records
    model_short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    task_name = args.task if custom_questions is None else "custom"
    output_file = create_output_file_path(task_name, args.method, model_short)
    logger.info(f"Question-answer records will be saved to: {output_file}")
    
    # Initialize progress bar manager
    # In multi-GPU mode (when gpu_id is set), disable tqdm progress bars
    # and output simple progress markers instead
    is_multi_gpu_worker = hasattr(args, 'gpu_id') and args.gpu_id is not None
    
    if is_multi_gpu_worker:
        # Multi-GPU worker: disable progress bars, use simple progress output
        progress_mgr = None
        logger.info(f"[Multi-GPU Worker] Running as GPU {args.gpu_id} worker - progress bars disabled")
    else:
        # Single GPU or orchestrator: use normal progress bars
        reset_progress_manager()
        progress_mgr = get_progress_manager()
    
    # Load configuration if using latent_mas_multipath
    multipath_config = None
    if args.method == "latent_mas_multipath":
        logger.info("[Configuration] Processing multi-path configuration")
        
        # Priority: config file > preset > command-line defaults
        if args.config:
            logger.info(f"[Configuration] Loading configuration from file: {args.config}")
            try:
                multipath_config = ConfigLoader.load_from_file(args.config)
                logger.info("[Configuration] Configuration file loaded successfully")
            except Exception as e:
                logger.error(f"[Configuration] Failed to load config file: {e}")
                raise
        elif args.config_preset:
            logger.info(f"[Configuration] Using preset configuration: {args.config_preset}")
            try:
                multipath_config = ConfigLoader.get_preset(args.config_preset)
                logger.info("[Configuration] Preset configuration loaded successfully")
            except Exception as e:
                logger.error(f"[Configuration] Failed to load preset: {e}")
                raise
        else:
            logger.debug("[Configuration] No config file or preset specified, using command-line arguments")
            # Create config from command-line arguments
            config_kwargs = {
                'num_paths': args.num_paths,
                'num_parent_paths': args.num_parent_paths,
                'enable_branching': args.enable_branching,
                'enable_merging': args.enable_merging,
                'pruning_strategy': args.pruning_strategy,
                'merge_threshold': args.merge_threshold,
                'branch_threshold': args.branch_threshold,
                'diversity_strategy': args.diversity_strategy,
                'latent_consistency_metric': args.latent_consistency_metric,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_new_tokens': args.max_new_tokens,
                'generate_bs': args.generate_bs,
                'enable_visualization': getattr(args, 'enable_visualization', True),
            }
            logger.debug(f"[Configuration] Using latent consistency metric: {args.latent_consistency_metric}")
            # Only include latent_steps if explicitly provided (not None)
            if args.latent_steps is not None:
                config_kwargs['latent_steps'] = args.latent_steps
            multipath_config = MultiPathConfig(**config_kwargs)
        
        # Merge with command-line arguments (CLI overrides config file/preset)
        if args.config or args.config_preset:
            logger.debug("[Configuration] Merging configuration with command-line arguments")
            multipath_config = ConfigLoader.merge_with_args(multipath_config, args)
        
        # Update args with final config values
        args.num_paths = multipath_config.num_paths
        args.num_parent_paths = multipath_config.num_parent_paths
        args.enable_branching = multipath_config.enable_branching
        args.enable_merging = multipath_config.enable_merging
        args.pruning_strategy = multipath_config.pruning_strategy
        args.merge_threshold = multipath_config.merge_threshold
        args.branch_threshold = multipath_config.branch_threshold
        args.diversity_strategy = multipath_config.diversity_strategy
        args.latent_consistency_metric = multipath_config.latent_consistency_metric
        args.latent_steps = multipath_config.latent_steps
        args.temperature = multipath_config.temperature
        args.top_p = multipath_config.top_p
        args.max_new_tokens = multipath_config.max_new_tokens
        args.generate_bs = multipath_config.generate_bs
        # Only update enable_visualization if not explicitly set via command line
        if not hasattr(args, 'enable_visualization') or args.enable_visualization is None:
            args.enable_visualization = multipath_config.enable_visualization
        
        logger.info(f"[Configuration] Final multi-path config: num_paths={args.num_paths}, "
                   f"num_parent_paths={args.num_parent_paths}, "
                   f"pruning={args.pruning_strategy}, diversity={args.diversity_strategy}, "
                   f"branching={args.enable_branching}, merging={args.enable_merging}, "
                   f"latent_consistency_metric={args.latent_consistency_metric}, "
                   f"latent_steps={args.latent_steps}, visualization={args.enable_visualization}")
    else:
        # For non-multipath methods, ensure enable_visualization attribute exists
        if not hasattr(args, 'enable_visualization'):
            args.enable_visualization = True
            logger.debug(f"[Visualization] Set default enable_visualization=True for {args.method} method")
    
    # Set default temperature if not specified
    if args.temperature is None:
        args.temperature = 0.7
        logger.info(f"[Temperature] No temperature specified, using default baseline temperature: {args.temperature}")
    else:
        logger.info(f"[Temperature] Using baseline temperature: {args.temperature}")
    
    # If custom_questions is provided, use it instead of dataset
    if custom_questions is not None:
        logger.info("Custom questions mode enabled")
        if args.max_samples == -1:
            args.max_samples = len(custom_questions)
        else:
            custom_questions = custom_questions[:args.max_samples]
            logger.info(f"Limited to {args.max_samples} questions")
    
    if args.method in ["latent_mas", "latent_mas_multipath"] and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
        logger.debug(f"Enabled vLLM-specific settings for {args.method}")
    
    logger.info(f"Initializing with method={args.method}, model={args.model_name}, seed={args.seed}")
    set_seed(args.seed)
    device = auto_device(args.device)
    logger.debug(f"Using device: {device}")
    
    logger.info("Loading model...")
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    logger.info("Model loaded successfully")
    
    start_time = time.time()

    # Log baseline temperature usage
    logger.info("=" * 80)
    logger.info(f"[Temperature Configuration]")
    logger.info(f"  Baseline temperature: {args.temperature}")
    logger.info(f"  This baseline will be used to generate a series of temperatures")
    logger.info(f"  for diversity strategies in multi-path reasoning")
    logger.info(f"  Temperature range: [{args.temperature - 0.3:.2f}, {args.temperature + 0.3:.2f}]")
    logger.info("=" * 80)
    
    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    logger.info(f"Initializing {args.method} method...")
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        # Use default latent_steps=10 if not specified
        latent_steps = args.latent_steps if args.latent_steps is not None else 10
        method = LatentMASMethod(
            model,
            latent_steps=latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )
    elif args.method == 'latent_mas_multipath':
        logger.info(f"[Method Init] Initializing LatentMASMultiPathMethod with latent_consistency_metric={args.latent_consistency_metric}")
        # Log topk_k parameter if pruning_strategy is topk
        if args.pruning_strategy == "topk":
            logger.info(f"[Method Init] Using topk pruning strategy with k={args.topk_k}")
        method = LatentMASMultiPathMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
            num_paths=args.num_paths,
            num_parent_paths=args.num_parent_paths,
            enable_branching=args.enable_branching,
            enable_merging=args.enable_merging,
            pruning_strategy=args.pruning_strategy,
            topk_k=args.topk_k,
            merge_threshold=args.merge_threshold,
            branch_threshold=args.branch_threshold,
            diversity_strategy=args.diversity_strategy,
            latent_consistency_metric=args.latent_consistency_metric,
        )
        
        # Enable PRM data collection if requested
        if getattr(args, 'collect_prm_data', False):
            logger.info("=" * 80)
            logger.info("[PRM Data Collection] Enabling PRM training data collection mode")
            logger.info("=" * 80)
            method.enable_prm_data_collection(
                output_dir=getattr(args, 'prm_output_dir', 'output/prm_data'),
                disable_pruning=getattr(args, 'prm_disable_pruning', True),
                disable_merging=getattr(args, 'prm_disable_merging', True)
            )
            logger.info(f"[PRM Data Collection] Output directory: {getattr(args, 'prm_output_dir', 'output/prm_data')}")
            logger.info(f"[PRM Data Collection] Pruning disabled: {getattr(args, 'prm_disable_pruning', True)}")
            logger.info(f"[PRM Data Collection] Merging disabled: {getattr(args, 'prm_disable_merging', True)}")
            logger.info("=" * 80)
    
    logger.info(f"Method {args.method} initialized successfully")

    # If custom questions provided, run on them directly
    if custom_questions is not None:
        # Create progress bar for custom questions (only if not in multi-GPU worker mode)
        if progress_mgr is not None:
            progress_mgr.create_main_progress(
                total=args.max_samples,
                desc=f"Processing custom questions",
                unit="question"
            )
        preds = run_custom_questions(method, custom_questions, args, progress_mgr, output_file)
    else:
        # dataset loading
        logger.info(f"Loading dataset: {args.task} (split: {args.split})")
        if args.task == "gsm8k":
            dataset_iter = load_gsm8k(split=args.split)
        elif args.task == "aime2024":
            dataset_iter = load_aime2024(split="train")
        elif args.task == "aime2025":
            dataset_iter = load_aime2025(split='train')
        elif args.task == "gpqa":
            dataset_iter = load_gpqa_diamond(split='test')
        elif args.task == "arc_easy":
            dataset_iter = load_arc_easy(split='test')
        elif args.task == "arc_challenge":
            dataset_iter = load_arc_challenge(split='test')
        elif args.task == "mbppplus":
            dataset_iter = load_mbppplus(split='test')
        elif args.task == "humanevalplus":
            dataset_iter = load_humanevalplus(split='test')
        elif args.task == "medqa":
            dataset_iter = load_medqa(split='test')
        else:
            raise ValueError(f'no {args.task} support')

        # Convert iterator to list for slicing support
        dataset_list = list(dataset_iter)
        total_dataset_size = len(dataset_list)
        logger.info(f"Loaded dataset with {total_dataset_size} total samples")
        
        # Handle multi-GPU data slicing
        if hasattr(args, 'data_start_idx') and args.data_start_idx > 0:
            logger.info(f"[Multi-GPU] This GPU will start from index {args.data_start_idx}")
            logger.info(f"[Multi-GPU] This GPU will process {args.max_samples} samples")
            
            # Slice dataset for this GPU
            end_idx = args.data_start_idx + args.max_samples
            dataset_list = dataset_list[args.data_start_idx:end_idx]
            
            logger.info(f"[Multi-GPU] Sliced dataset: [{args.data_start_idx}:{end_idx}] ({len(dataset_list)} samples)")
        else:
            # Single GPU mode
            if args.max_samples == -1:
                args.max_samples = total_dataset_size
                logger.info(f"Will process all {args.max_samples} samples from dataset")
            else:
                dataset_list = dataset_list[:args.max_samples]
                logger.info(f"Will process {args.max_samples} samples from dataset")

        preds: List[Dict] = []
        processed = 0
        batch: List[Dict] = []

        # Create progress bar (only if not in multi-GPU worker mode)
        if progress_mgr is not None:
            progress_mgr.create_main_progress(
                total=args.max_samples,
                desc=f"Processing {args.task}",
                unit="sample"
            )

        for item in dataset_list:
            if processed >= args.max_samples:
                break
            batch.append(item)
            if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
                processed, preds = process_batch(
                    method,
                    batch,
                    processed,
                    preds,
                    args.max_samples,
                    args,
                    progress_mgr,
                    output_file,
                )
                batch = []
                if processed >= args.max_samples:
                    break

        if batch and processed < args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                max_samples=args.max_samples,
                args=args,
                progress_mgr=progress_mgr,
                output_file=output_file,
            )
        logger.info(f"Completed processing {len(preds)} samples from dataset")
    
    # Close progress bar
    if progress_mgr is not None:
        progress_mgr.close_all()
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")

    acc, correct = evaluate(preds)
    logger.info(f"Final results: {correct}/{len(preds)} correct, accuracy: {acc:.4f}")
    
    # Load results in JSON format
    result_json = json.dumps(
        {
            "method": args.method,
            "model": args.model_name,
            "split": args.split if custom_questions is None else "custom",
            "seed": args.seed,
            "max_samples": args.max_samples,
            "accuracy": acc,
            "correct": correct,
            "total_time_sec": round(total_time,4),
            "time_per_sample_sec": round(total_time / args.max_samples, 4) if args.max_samples > 0 else 0,
        },
        ensure_ascii=False,
    )
    
    # Print final results to console
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(result_json)
    print("="*80)
    logger.debug(f"Result JSON: {result_json}")
    
    # Save run parameters and results to log file
    save_run_result_log(
        args=args,
        total_time=total_time,
        acc=acc,
        correct=correct,
        total=len(preds),
        custom_questions=custom_questions
    )
    
    # Save results to CSV file
    logger.info("=" * 80)
    logger.info("Saving results to CSV file")
    logger.info("=" * 80)
    
    try:
        # Prepare run parameters dictionary
        task_name = args.task if custom_questions is None else "custom"
        run_params = {
            # Core parameters
            "method": args.method,
            "model_name": args.model_name,
            "task": task_name,
            "split": args.split if custom_questions is None else "custom",
            "max_samples": args.max_samples,
            "seed": args.seed,
            "device": args.device,
            "prompt": getattr(args, 'prompt', 'sequential'),
            
            # Generation parameters
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "generate_bs": args.generate_bs,
            
            # Method-specific parameters
            "latent_steps": getattr(args, 'latent_steps', None),
            "text_mas_context_length": getattr(args, 'text_mas_context_length', -1),
            "think": getattr(args, 'think', False),
            "latent_space_realign": getattr(args, 'latent_space_realign', False),
            
            # vLLM parameters
            "use_vllm": getattr(args, 'use_vllm', False),
            "enable_prefix_caching": getattr(args, 'enable_prefix_caching', False),
            "use_second_HF_model": getattr(args, 'use_second_HF_model', False),
            "device2": getattr(args, 'device2', None),
            "tensor_parallel_size": getattr(args, 'tensor_parallel_size', 1),
            "gpu_memory_utilization": getattr(args, 'gpu_memory_utilization', 0.9),
            
            # Multi-path specific parameters (if applicable)
            "num_paths": getattr(args, 'num_paths', None),
            "enable_branching": getattr(args, 'enable_branching', None),
            "enable_merging": getattr(args, 'enable_merging', None),
            "pruning_strategy": getattr(args, 'pruning_strategy', None),
            "merge_threshold": getattr(args, 'merge_threshold', None),
            "branch_threshold": getattr(args, 'branch_threshold', None),
            "diversity_strategy": getattr(args, 'diversity_strategy', None),
            "latent_consistency_metric": getattr(args, 'latent_consistency_metric', None),
            
            # Configuration file parameters
            "config": getattr(args, 'config', None),
            "config_preset": getattr(args, 'config_preset', None),
            
            # Visualization
            "enable_visualization": getattr(args, 'enable_visualization', True),
        }
        
        # Remove None values for cleaner output
        run_params = {k: v for k, v in run_params.items() if v is not None}
        
        # Prepare results dictionary
        results = {
            "accuracy": acc,
            "success_rate": acc,  # Same as accuracy
            "correct": correct,
            "total": len(preds),
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / len(preds), 4) if len(preds) > 0 else 0.0,
        }
        
        # Save to CSV
        save_to_csv_results(
            run_params=run_params,
            results=results,
            timestamp=datetime.now()
        )
        
        logger.info("Results successfully saved to CSV file: output/csv_res/results.csv")
        
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}", exc_info=True)
        logger.debug(f"CSV save error: {type(e).__name__}: {str(e)}")
    
    # Save collected PRM training data if enabled
    if getattr(args, 'collect_prm_data', False) and hasattr(method, 'prm_data_collector'):
        logger.info("=" * 80)
        logger.info("[PRM Data Collection] Saving collected PRM training data")
        logger.info("=" * 80)
        
        try:
            # Get collected data from the data collector
            collected_data = method.prm_data_collector.get_collected_data()
            logger.info(f"[PRM Data Collection] Retrieved {len(collected_data)} question records")
            
            if len(collected_data) == 0:
                logger.warning("[PRM Data Collection] No data collected - nothing to save")
            else:
                # Get statistics
                stats = method.prm_data_collector.get_statistics()
                logger.info(f"[PRM Data Collection] Statistics:")
                logger.info(f"  - Total questions: {stats.get('total_questions', 0)}")
                logger.info(f"  - Total paths: {stats.get('total_paths', 0)}")
                logger.info(f"  - Correct questions: {stats.get('correct_questions', 0)}")
                logger.info(f"  - Accuracy: {stats.get('accuracy', 0.0):.4f}")
                logger.info(f"  - Avg paths per question: {stats.get('avg_paths_per_question', 0.0):.2f}")
                
                # Build tree structures for each question
                logger.info("[PRM Data Collection] Building tree structures for collected data")
                tree_structures = []
                for idx, question_record in enumerate(collected_data):
                    logger.debug(f"[PRM Data Collection] Building tree for question {idx + 1}/{len(collected_data)}")
                    logger.debug(f"[PRM Data Collection] Question has {len(question_record.paths)} paths")
                    
                    tree_structure = method.prm_tree_builder.build_tree(
                        path_records=question_record.paths,
                        is_correct=question_record.is_correct
                    )
                    tree_structures.append(tree_structure)
                    
                    # Log tree structure details
                    num_nodes = tree_structure.get('num_nodes', 0)
                    num_edges = tree_structure.get('num_edges', 0)
                    max_depth = tree_structure.get('max_depth', 0)
                    logger.debug(f"[PRM Data Collection] Tree built: {num_nodes} nodes, "
                               f"{num_edges} edges, max_depth={max_depth}")
                    
                    if num_nodes == 0:
                        logger.warning(f"[PRM Data Collection] Question {idx + 1} has no paths - tree is empty")
                
                logger.info(f"[PRM Data Collection] Built {len(tree_structures)} tree structures")
                
                # Save data using PRMDataStorage
                logger.info("[PRM Data Collection] Saving data to disk")
                output_dir = getattr(args, 'prm_output_dir', 'prm_data')
                logger.info(f"[PRM Data Collection] Output directory: {output_dir}")
                
                # Create batch name based on task and timestamp
                task_name = args.task if custom_questions is None else "custom"
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Add output suffix for multi-GPU mode
                if hasattr(args, 'output_suffix') and args.output_suffix:
                    batch_name = f"{task_name}_{args.method}_{timestamp_str}_{args.output_suffix}"
                    logger.info(f"[PRM Data Collection] Multi-GPU mode: Using suffix '{args.output_suffix}'")
                else:
                    batch_name = f"{task_name}_{args.method}_{timestamp_str}"
                
                logger.info(f"[PRM Data Collection] Batch name: {batch_name}")
                
                # Save batch data
                saved_path = method.prm_data_storage.save_batch_data(
                    question_records=collected_data,
                    tree_structures=tree_structures,
                    batch_name=batch_name
                )
                logger.info(f"[PRM Data Collection] Data saved to: {saved_path}")
                
                # Save metadata
                metadata = {
                    "task": task_name,
                    "method": args.method,
                    "model": args.model_name,
                    "num_questions": len(collected_data),
                    "num_paths_total": stats.get('total_paths', 0),
                    "accuracy": stats.get('accuracy', 0.0),
                    "timestamp": datetime.now().isoformat(),
                    "batch_name": batch_name,
                    "data_file": saved_path,
                }
                metadata_path = method.prm_data_storage.save_metadata(
                    metadata=metadata,
                    filename=f"{batch_name}_metadata.json"
                )
                logger.info(f"[PRM Data Collection] Metadata saved to: {metadata_path}")
                
                # Get storage statistics
                storage_stats = method.prm_data_storage.get_statistics()
                logger.info("[PRM Data Collection] Storage statistics:")
                logger.info(f"  - Output directory: {storage_stats['output_dir']}")
                logger.info(f"  - Number of files: {storage_stats['num_files']}")
                logger.info(f"  - Total size: {storage_stats['total_size_mb']:.2f} MB")
                
                logger.info("=" * 80)
                logger.info("[PRM Data Collection] PRM training data saved successfully!")
                logger.info("=" * 80)
                logger.info(f"Data location: {saved_path}")
                logger.info(f"Metadata location: {metadata_path}")
                logger.info(f"Total questions: {len(collected_data)}")
                logger.info(f"Total paths: {stats.get('total_paths', 0)}")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"[PRM Data Collection] Failed to save PRM data: {e}", exc_info=True)
            logger.error(f"[PRM Data Collection] Error type: {type(e).__name__}")
            logger.error(f"[PRM Data Collection] Error message: {str(e)}")



if __name__ == "__main__":
    main()
