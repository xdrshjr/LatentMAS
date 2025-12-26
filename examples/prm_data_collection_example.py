"""Example script for using collected PRM training data.

This script demonstrates how to load and use the collected latent PRM
training data for model training or analysis.
"""

import torch
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> Dict[str, Any]:
    """Load data from a .pt file (supports both single question and batch formats).
    
    Args:
        filepath: Path to the .pt file
        
    Returns:
        Dictionary with data (batch format with 'questions' list)
    """
    logger.info(f"Loading data from: {filepath}\n")
    # Set weights_only=False since this is our own generated data
    # PyTorch 2.6+ defaults to weights_only=True for security
    data = torch.load(filepath, weights_only=False)
    
    # Check if this is a batch file or single question file
    if "batch_name" in data:
        # Batch format
        logger.info(f"Batch: {data['batch_name']}")
        logger.info(f"Number of questions: {data['num_questions']}\n")
        return data
    elif "question_id" in data:
        # Single question format - convert to batch format for consistency
        logger.info(f"Single question file\n")
        batch_data = {
            "batch_name": "single_question",
            "num_questions": 1,
            "questions": [data]
        }
        return batch_data
    else:
        raise ValueError(f"Unknown data format in {filepath}")


def print_question_summary(question_data: Dict[str, Any], question_idx: int) -> None:
    """Print summary of a single question.
    
    Args:
        question_data: Question data dictionary
        question_idx: Index of the question
    """
    logger.info(f"Question {question_idx + 1}:")
    logger.info(f"  ID: {question_data['question_id']}")
    logger.info(f"  Question: {question_data['question'][:100]}...")
    logger.info(f"  Gold Answer: {question_data['gold_answer']}")
    logger.info(f"  Final Answer: {question_data['final_answer']}")
    logger.info(f"  Correct: {question_data['is_correct']}")
    logger.info(f"  Number of paths: {len(question_data['paths'])}\n")


def display_path_details(question_data: Dict[str, Any], max_paths: int = 10) -> None:
    """Display detailed path information with scores.
    
    Args:
        question_data: Question data dictionary
        max_paths: Maximum number of paths to display in detail
    """
    logger.info("=" * 80)
    logger.info("PATH DETAILS WITH SCORES")
    logger.info("=" * 80)
    
    paths = question_data['paths']
    
    if not paths:
        logger.info("No paths found.\n")
        return
    
    # Group paths by agent
    paths_by_agent = {}
    for path in paths:
        agent_name = path['agent_name']
        if agent_name not in paths_by_agent:
            paths_by_agent[agent_name] = []
        paths_by_agent[agent_name].append(path)
    
    logger.info(f"Total paths: {len(paths)}")
    logger.info(f"Agents: {list(paths_by_agent.keys())}\n")
    
    # Display paths by agent
    for agent_name in sorted(paths_by_agent.keys()):
        agent_paths = paths_by_agent[agent_name]
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Agent: {agent_name}")
        logger.info(f"{'─' * 80}")
        logger.info(f"Number of paths: {len(agent_paths)}\n")
        
        # Sort paths by PRM score (descending) for better visualization
        sorted_paths = sorted(
            agent_paths, 
            key=lambda p: p['prm_score'] if p['prm_score'] is not None else -1,
            reverse=True
        )
        
        # Display up to max_paths
        display_count = min(len(sorted_paths), max_paths)
        
        for i, path in enumerate(sorted_paths[:display_count], 1):
            # Build path traversal string
            traversal = _build_path_traversal_string(path, paths)
            
            logger.info(f"  [{i}] Path ID: {path['path_id']}")
            logger.info(f"      Traversal: {traversal}")
            logger.info(f"      Quality Score: {path['score']:.4f}")
            
            # Display PRM score with clear formatting
            if path['prm_score'] is not None:
                prm_score = path['prm_score']
                score_bar = _create_score_bar(prm_score)
                logger.info(f"      PRM Score: {prm_score:.4f} {score_bar}")
            else:
                logger.info(f"      PRM Score: None (not computed)")
            
            logger.info(f"      Latent Steps: {path['num_latent_steps']}")
            logger.info(f"      Parent: {path['parent_path_id'] if path['parent_path_id'] is not None else 'None (root)'}")
            logger.info(f"      Children: {len(path['child_path_ids'])} paths")
            
            # Display individual correctness if available
            if 'is_correct' in path['metadata']:
                is_correct = path['metadata']['is_correct']
                logger.info(f"      Individual Correctness: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            
            logger.info("")
        
        if len(sorted_paths) > display_count:
            logger.info(f"  ... and {len(sorted_paths) - display_count} more paths\n")
        
        # Display PRM score statistics for this agent
        prm_scores = [p['prm_score'] for p in agent_paths if p['prm_score'] is not None]
        if prm_scores:
            logger.info(f"  PRM Score Statistics:")
            logger.info(f"    Min: {min(prm_scores):.4f}")
            logger.info(f"    Max: {max(prm_scores):.4f}")
            logger.info(f"    Mean: {np.mean(prm_scores):.4f}")
            logger.info(f"    Std: {np.std(prm_scores):.4f}")
        else:
            logger.info(f"  PRM Score Statistics: No scores available")


def _build_path_traversal_string(path: Dict[str, Any], all_paths: List[Dict[str, Any]]) -> str:
    """Build a string showing the path traversal from root to this path.
    
    Args:
        path: Current path dictionary
        all_paths: List of all paths
        
    Returns:
        Formatted traversal string
    """
    # Build path_id to path mapping
    path_map = {p['path_id']: p for p in all_paths}
    
    # Traverse backwards to root
    traversal = []
    current_id = path['path_id']
    visited = set()
    
    while current_id is not None and current_id not in visited:
        visited.add(current_id)
        if current_id in path_map:
            current_path = path_map[current_id]
            traversal.insert(0, f"path_{current_id}({current_path['agent_name']})")
            current_id = current_path['parent_path_id']
        else:
            break
    
    return " → ".join(traversal) if traversal else f"path_{path['path_id']} (isolated)"


def _create_score_bar(score: float, width: int = 20) -> str:
    """Create a visual bar representation of a score.
    
    Args:
        score: Score value between 0 and 1
        width: Width of the bar in characters
        
    Returns:
        String representation of the score bar
    """
    filled = int(score * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}]"


def analyze_tree_structure(question_data: Dict[str, Any]) -> None:
    """Analyze the tree structure.
    
    Args:
        question_data: Question data dictionary
    """
    tree = question_data['tree_structure']
    
    logger.info("\n" + "=" * 80)
    logger.info("TREE STRUCTURE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Number of nodes: {tree['num_nodes']}")
    logger.info(f"Number of edges: {tree['num_edges']}")
    logger.info(f"Max depth: {tree['max_depth']}")
    logger.info(f"Root nodes: {tree['root_ids']}")
    logger.info(f"Final answer correct: {tree['is_correct']}\n")
    
    # Analyze nodes by depth
    nodes_by_depth = {}
    for node in tree['nodes']:
        depth = node['depth']
        if depth not in nodes_by_depth:
            nodes_by_depth[depth] = []
        nodes_by_depth[depth].append(node)
    
    logger.info("Nodes by depth:")
    for depth in sorted(nodes_by_depth.keys()):
        nodes = nodes_by_depth[depth]
        logger.info(f"  Depth {depth}: {len(nodes)} nodes")
        
        # Show PRM score statistics
        prm_scores = [n['prm_score'] for n in nodes if n['prm_score'] is not None]
        if prm_scores:
            logger.info(f"    PRM scores: min={min(prm_scores):.4f}, "
                      f"max={max(prm_scores):.4f}, mean={np.mean(prm_scores):.4f}")
        else:
            logger.info(f"    PRM scores: None available")


def extract_training_samples(question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract training samples from collected data.
    
    Args:
        question_data: Question data dictionary
        
    Returns:
        List of training samples
    """
    samples = []
    
    for path in question_data['paths']:
        if path['prm_score'] is not None:
            sample = {
                'latent_sequence': path['latent_history'],  # [num_steps, hidden_dim]
                'label': path['prm_score'],  # Target score
                'metadata': {
                    'path_id': path['path_id'],
                    'agent_name': path['agent_name'],
                    'agent_idx': path['agent_idx'],
                    'question_correct': question_data['is_correct'],
                }
            }
            samples.append(sample)
    
    return samples


def extract_all_training_samples(batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract training samples from all questions in batch.
    
    Args:
        batch_data: Batch data dictionary with 'questions' list
        
    Returns:
        List of training samples from all questions
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTING TRAINING SAMPLES")
    logger.info("=" * 80)
    
    all_samples = []
    
    for question_data in batch_data['questions']:
        samples = extract_training_samples(question_data)
        all_samples.extend(samples)
    
    if all_samples:
        logger.info(f"Extracted {len(all_samples)} training samples from {batch_data['num_questions']} questions")
        logger.info(f"Sample latent shape: {all_samples[0]['latent_sequence'].shape}")
        
        labels = [s['label'] for s in all_samples]
        logger.info(f"Label statistics:")
        logger.info(f"  Min: {min(labels):.4f}")
        logger.info(f"  Max: {max(labels):.4f}")
        logger.info(f"  Mean: {np.mean(labels):.4f}")
        logger.info(f"  Std: {np.std(labels):.4f}")
    else:
        logger.warning("No training samples found (all PRM scores are None)")
    
    logger.info("")
    return all_samples


def create_pytorch_dataset(samples: List[Dict[str, Any]]) -> torch.utils.data.Dataset:
    """Create a PyTorch dataset from training samples.
    
    Args:
        samples: List of training samples
        
    Returns:
        PyTorch Dataset
    """
    class LatentPRMDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            return {
                'latent_sequence': sample['latent_sequence'],
                'label': torch.tensor(sample['label'], dtype=torch.float32),
                'metadata': sample['metadata']
            }
    
    dataset = LatentPRMDataset(samples)
    logger.info(f"Created PyTorch dataset with {len(dataset)} samples\n")
    return dataset


def main():
    """Main function to demonstrate data loading and analysis."""
    
    # Example: Load data from prm_data directory
    data_dir = Path("prm_data")
    
    # Find the first .pt file
    pt_files = list(data_dir.glob("*.pt"))
    
    if not pt_files:
        logger.error("No .pt files found in ./prm_data/")
        logger.info("Please run collect_training_data.sh first to collect data.")
        return
    
    # Load first file (supports both single question and batch formats)
    filepath = pt_files[0]
    batch_data = load_data(str(filepath))
    
    logger.info("=" * 80)
    logger.info("DATA OVERVIEW")
    logger.info("=" * 80)
    logger.info(f"Total questions: {batch_data['num_questions']}\n")
    
    # Analyze first question in detail
    first_question = batch_data['questions'][0]
    print_question_summary(first_question, 0)
    
    # Display detailed path information with scores
    display_path_details(first_question, max_paths=10)
    
    # Analyze tree structure
    analyze_tree_structure(first_question)
    
    # Extract training samples from all questions
    samples = extract_all_training_samples(batch_data)
    
    if not samples:
        logger.error("No training samples found. Cannot create dataset.")
        logger.info("\nThis may indicate that PRM scores were not computed during data collection.")
        logger.info("Please check the data collection logs for any errors.")
        return
    
    # Create PyTorch dataset
    dataset = create_pytorch_dataset(samples)
    
    # Example: Access a sample
    logger.info("=" * 80)
    logger.info("EXAMPLE SAMPLE")
    logger.info("=" * 80)
    sample = dataset[0]
    logger.info(f"Latent sequence shape: {sample['latent_sequence'].shape}")
    logger.info(f"Label: {sample['label'].item():.4f}")
    logger.info(f"Metadata: {sample['metadata']}\n")
    
    logger.info("=" * 80)
    logger.info("DATA LOADING AND ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Create a DataLoader for batch training")
    logger.info("2. Define a PRM model architecture")
    logger.info("3. Train the model using the collected data")
    logger.info("4. Evaluate on validation set")


if __name__ == "__main__":
    main()
