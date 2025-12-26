"""Example script for using collected PRM training data.

This script demonstrates how to load and use the collected latent PRM
training data for model training or analysis.
"""

import torch
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any


def load_data(filepath: str) -> Dict[str, Any]:
    """Load data from a .pt file (supports both single question and batch formats).
    
    Args:
        filepath: Path to the .pt file
        
    Returns:
        Dictionary with data (batch format with 'questions' list)
    """
    print(f"Loading data from: {filepath}")
    # Set weights_only=False since this is our own generated data
    # PyTorch 2.6+ defaults to weights_only=True for security
    data = torch.load(filepath, weights_only=False)
    
    # Check if this is a batch file or single question file
    if "batch_name" in data:
        # Batch format
        print(f"Batch: {data['batch_name']}")
        print(f"Number of questions: {data['num_questions']}")
        print()
        return data
    elif "question_id" in data:
        # Single question format - convert to batch format for consistency
        print(f"Single question file")
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
    print(f"Question {question_idx + 1}:")
    print(f"  ID: {question_data['question_id']}")
    print(f"  Question: {question_data['question'][:100]}...")
    print(f"  Gold Answer: {question_data['gold_answer']}")
    print(f"  Final Answer: {question_data['final_answer']}")
    print(f"  Correct: {question_data['is_correct']}")
    print(f"  Number of paths: {len(question_data['paths'])}")
    print()


def analyze_path_tree(question_data: Dict[str, Any]) -> None:
    """Analyze the path tree structure.
    
    Args:
        question_data: Question data dictionary
    """
    tree = question_data['tree_structure']
    
    print("=" * 80)
    print("Tree Structure Analysis")
    print("=" * 80)
    print(f"Number of nodes: {tree['num_nodes']}")
    print(f"Number of edges: {tree['num_edges']}")
    print(f"Max depth: {tree['max_depth']}")
    print(f"Root nodes: {tree['root_ids']}")
    print(f"Final answer correct: {tree['is_correct']}")
    print()
    
    # Analyze nodes by depth
    nodes_by_depth = {}
    for node in tree['nodes']:
        depth = node['depth']
        if depth not in nodes_by_depth:
            nodes_by_depth[depth] = []
        nodes_by_depth[depth].append(node)
    
    print("Nodes by depth:")
    for depth in sorted(nodes_by_depth.keys()):
        nodes = nodes_by_depth[depth]
        print(f"  Depth {depth}: {len(nodes)} nodes")
        
        # Show PRM score statistics
        prm_scores = [n['prm_score'] for n in nodes if n['prm_score'] is not None]
        if prm_scores:
            print(f"    PRM scores: min={min(prm_scores):.4f}, "
                  f"max={max(prm_scores):.4f}, mean={np.mean(prm_scores):.4f}")
    print()


def analyze_paths(question_data: Dict[str, Any]) -> None:
    """Analyze individual paths.
    
    Args:
        question_data: Question data dictionary
    """
    print("=" * 80)
    print("Path Analysis")
    print("=" * 80)
    
    paths = question_data['paths']
    
    # Group paths by agent
    paths_by_agent = {}
    for path in paths:
        agent_name = path['agent_name']
        if agent_name not in paths_by_agent:
            paths_by_agent[agent_name] = []
        paths_by_agent[agent_name].append(path)
    
    print(f"Total paths: {len(paths)}")
    print(f"Agents: {list(paths_by_agent.keys())}")
    print()
    
    for agent_name, agent_paths in paths_by_agent.items():
        print(f"Agent: {agent_name}")
        print(f"  Number of paths: {len(agent_paths)}")
        
        # PRM score statistics
        prm_scores = [p['prm_score'] for p in agent_paths if p['prm_score'] is not None]
        if prm_scores:
            print(f"  PRM scores: min={min(prm_scores):.4f}, "
                  f"max={max(prm_scores):.4f}, mean={np.mean(prm_scores):.4f}")
        
        # Latent vector statistics
        latent_shapes = [p['latent_history'].shape for p in agent_paths]
        print(f"  Latent shapes: {latent_shapes[0]} (example)")
        print()


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
    print("=" * 80)
    print("Extracting Training Samples from All Questions")
    print("=" * 80)
    
    all_samples = []
    
    for question_data in batch_data['questions']:
        samples = extract_training_samples(question_data)
        all_samples.extend(samples)
    
    if all_samples:
        print(f"Extracted {len(all_samples)} training samples from {batch_data['num_questions']} questions")
        print(f"Sample latent shape: {all_samples[0]['latent_sequence'].shape}")
        print(f"Label range: [{min(s['label'] for s in all_samples):.4f}, "
              f"{max(s['label'] for s in all_samples):.4f}]")
    else:
        print("No training samples found (all PRM scores are None)")
    print()
    
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
    print(f"Created PyTorch dataset with {len(dataset)} samples")
    return dataset


def main():
    """Main function to demonstrate data loading and analysis."""
    
    # Example: Load data from prm_data directory
    data_dir = Path("prm_data")
    
    # Find the first .pt file
    pt_files = list(data_dir.glob("*.pt"))
    
    if not pt_files:
        print("No .pt files found in ./prm_data/")
        print("Please run collect_training_data.sh first to collect data.")
        return
    
    # Load first file (supports both single question and batch formats)
    filepath = pt_files[0]
    batch_data = load_data(str(filepath))
    
    print("=" * 80)
    print("Data Overview")
    print("=" * 80)
    print(f"Total questions: {batch_data['num_questions']}")
    print()
    
    # Analyze first question in detail
    first_question = batch_data['questions'][0]
    print_question_summary(first_question, 0)
    
    # Analyze the first question's tree and paths
    analyze_path_tree(first_question)
    analyze_paths(first_question)
    
    # Extract training samples from all questions
    samples = extract_all_training_samples(batch_data)
    
    if not samples:
        print("No training samples found. Cannot create dataset.")
        return
    
    # Create PyTorch dataset
    dataset = create_pytorch_dataset(samples)
    
    # Example: Access a sample
    print("=" * 80)
    print("Example Sample")
    print("=" * 80)
    sample = dataset[0]
    print(f"Latent sequence shape: {sample['latent_sequence'].shape}")
    print(f"Label: {sample['label'].item():.4f}")
    print(f"Metadata: {sample['metadata']}")
    print()
    
    print("=" * 80)
    print("Data loading and analysis complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Create a DataLoader for batch training")
    print("2. Define a PRM model architecture")
    print("3. Train the model using the collected data")
    print("4. Evaluate on validation set")


if __name__ == "__main__":
    main()

