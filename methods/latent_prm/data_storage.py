"""Data storage module for saving collected PRM training data.

This module handles saving collected path data in PyTorch .pt format
for efficient loading during PRM model training.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class PRMDataStorage:
    """Storage handler for PRM training data.
    
    Saves collected data in PyTorch .pt format with accompanying metadata.
    
    Attributes:
        output_dir: Directory to save data files
    """
    
    def __init__(self, output_dir: str = "prm_data"):
        """Initialize the data storage handler.
        
        Args:
            output_dir: Directory to save data files (default: prm_data at project root)
        """
        self.output_dir = Path(output_dir)
        logger.info(f"[PRMDataStorage] Initializing with output_dir: {self.output_dir}")
        logger.debug(f"[PRMDataStorage] Absolute path: {self.output_dir.resolve()}")
        
        # Create directory if it doesn't exist
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[PRMDataStorage] Output directory created/verified: {self.output_dir}")
            logger.debug(f"[PRMDataStorage] Directory exists: {self.output_dir.exists()}")
            logger.debug(f"[PRMDataStorage] Directory is writable: {self.output_dir.is_dir()}")
        except Exception as e:
            logger.error(f"[PRMDataStorage] Failed to create output directory: {e}", exc_info=True)
            raise
    
    def save_question_data(
        self,
        question_record: Any,
        tree_structure: Dict[str, Any],
        question_idx: int
    ) -> str:
        """Save data for a single question.
        
        Args:
            question_record: QuestionRecord object
            tree_structure: Tree structure dictionary from PathTreeBuilder
            question_idx: Index of the question
            
        Returns:
            Path to the saved file
        """
        logger.info(f"[PRMDataStorage] Saving question {question_idx}: {question_record.question_id}")
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"question_{question_idx:06d}_{timestamp}.pt"
        filepath = self.output_dir / filename
        
        # Prepare data for saving
        data = {
            "question_id": question_record.question_id,
            "question": question_record.question,
            "gold_answer": question_record.gold_answer,
            "final_answer": question_record.final_answer,
            "is_correct": question_record.is_correct,
            "tree_structure": tree_structure,
            "paths": [],
            "metadata": {
                "question_idx": question_idx,
                "timestamp": timestamp,
                "num_paths": len(question_record.paths),
            }
        }
        
        # Process each path
        for path_record in question_record.paths:
            path_data = {
                "path_id": path_record.path_id,
                "agent_name": path_record.agent_name,
                "agent_idx": path_record.agent_idx,
                "parent_path_id": path_record.parent_path_id,
                "child_path_ids": path_record.child_path_ids,
                "score": path_record.score,
                "prm_score": path_record.prm_score,
                "metadata": path_record.metadata,
                # Convert latent tensors to a single stacked tensor
                "latent_history": torch.stack(path_record.latent_history) if path_record.latent_history else None,
                "hidden_states": path_record.hidden_states,
                "num_latent_steps": len(path_record.latent_history),
            }
            data["paths"].append(path_data)
        
        # Save to .pt file
        try:
            torch.save(data, filepath)
            logger.info(f"[PRMDataStorage] Saved question data to: {filepath}")
            logger.debug(f"[PRMDataStorage] File size: {filepath.stat().st_size / 1024:.2f} KB")
            return str(filepath)
        except Exception as e:
            logger.error(f"[PRMDataStorage] Failed to save question data: {e}", exc_info=True)
            raise
    
    def save_batch_data(
        self,
        question_records: List[Any],
        tree_structures: List[Dict[str, Any]],
        batch_name: str = "batch"
    ) -> str:
        """Save data for a batch of questions.
        
        Args:
            question_records: List of QuestionRecord objects
            tree_structures: List of tree structure dictionaries
            batch_name: Name for this batch
            
        Returns:
            Path to the saved file
        """
        logger.info(f"[PRMDataStorage] Saving batch '{batch_name}' with {len(question_records)} questions")
        
        if len(question_records) != len(tree_structures):
            raise ValueError(
                f"Mismatch: {len(question_records)} questions but {len(tree_structures)} tree structures"
            )
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{batch_name}_{timestamp}.pt"
        filepath = self.output_dir / filename
        
        # Prepare batch data
        batch_data = {
            "batch_name": batch_name,
            "timestamp": timestamp,
            "num_questions": len(question_records),
            "questions": []
        }
        
        # Process each question
        for idx, (question_record, tree_structure) in enumerate(zip(question_records, tree_structures)):
            question_data = {
                "question_id": question_record.question_id,
                "question": question_record.question,
                "gold_answer": question_record.gold_answer,
                "final_answer": question_record.final_answer,
                "is_correct": question_record.is_correct,
                "tree_structure": tree_structure,
                "paths": []
            }
            
            # Process each path
            for path_record in question_record.paths:
                path_data = {
                    "path_id": path_record.path_id,
                    "agent_name": path_record.agent_name,
                    "agent_idx": path_record.agent_idx,
                    "parent_path_id": path_record.parent_path_id,
                    "child_path_ids": path_record.child_path_ids,
                    "score": path_record.score,
                    "prm_score": path_record.prm_score,
                    "metadata": path_record.metadata,
                    "latent_history": torch.stack(path_record.latent_history) if path_record.latent_history else None,
                    "hidden_states": path_record.hidden_states,
                    "num_latent_steps": len(path_record.latent_history),
                }
                question_data["paths"].append(path_data)
            
            batch_data["questions"].append(question_data)
        
        # Save to .pt file
        try:
            logger.debug(f"[PRMDataStorage] Saving batch data to: {filepath}")
            logger.debug(f"[PRMDataStorage] Batch contains {len(batch_data['questions'])} questions")
            
            torch.save(batch_data, filepath)
            
            # Verify file was created
            if not filepath.exists():
                raise IOError(f"File was not created: {filepath}")
            
            file_size_mb = filepath.stat().st_size / (1024*1024)
            logger.info(f"[PRMDataStorage] ✓ Successfully saved batch data to: {filepath}")
            logger.info(f"[PRMDataStorage] File size: {file_size_mb:.2f} MB")
            logger.info(f"[PRMDataStorage] Absolute path: {filepath.resolve()}")
            
            return str(filepath)
        except Exception as e:
            logger.error(f"[PRMDataStorage] ✗ Failed to save batch data to {filepath}: {e}", exc_info=True)
            logger.error(f"[PRMDataStorage] Output directory: {self.output_dir}")
            logger.error(f"[PRMDataStorage] Output directory exists: {self.output_dir.exists()}")
            raise
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = "metadata.json"
    ) -> str:
        """Save metadata about the collected data.
        
        Args:
            metadata: Metadata dictionary
            filename: Name of the metadata file
            
        Returns:
            Path to the saved file
        """
        logger.info(f"[PRMDataStorage] Saving metadata to: {filename}")
        
        filepath = self.output_dir / filename
        
        try:
            logger.debug(f"[PRMDataStorage] Saving metadata to: {filepath}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Verify file was created
            if not filepath.exists():
                raise IOError(f"Metadata file was not created: {filepath}")
            
            file_size_kb = filepath.stat().st_size / 1024
            logger.info(f"[PRMDataStorage] ✓ Successfully saved metadata to: {filepath}")
            logger.info(f"[PRMDataStorage] Metadata file size: {file_size_kb:.2f} KB")
            logger.info(f"[PRMDataStorage] Absolute path: {filepath.resolve()}")
            
            return str(filepath)
        except Exception as e:
            logger.error(f"[PRMDataStorage] ✗ Failed to save metadata to {filepath}: {e}", exc_info=True)
            raise
    
    def load_question_data(self, filepath: str) -> Dict[str, Any]:
        """Load data for a single question.
        
        Args:
            filepath: Path to the .pt file
            
        Returns:
            Dictionary with question data
        """
        logger.info(f"[PRMDataStorage] Loading question data from: {filepath}")
        
        try:
            data = torch.load(filepath)
            logger.info(f"[PRMDataStorage] Loaded question {data['question_id']}")
            logger.debug(f"[PRMDataStorage] Question has {len(data['paths'])} paths")
            return data
        except Exception as e:
            logger.error(f"[PRMDataStorage] Failed to load question data: {e}", exc_info=True)
            raise
    
    def load_batch_data(self, filepath: str) -> Dict[str, Any]:
        """Load data for a batch of questions.
        
        Args:
            filepath: Path to the .pt file
            
        Returns:
            Dictionary with batch data
        """
        logger.info(f"[PRMDataStorage] Loading batch data from: {filepath}")
        
        try:
            data = torch.load(filepath)
            logger.info(f"[PRMDataStorage] Loaded batch '{data['batch_name']}' "
                       f"with {data['num_questions']} questions")
            return data
        except Exception as e:
            logger.error(f"[PRMDataStorage] Failed to load batch data: {e}", exc_info=True)
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about saved data files.
        
        Returns:
            Dictionary with statistics
        """
        logger.info("[PRMDataStorage] Computing statistics")
        
        # Find all .pt files
        pt_files = list(self.output_dir.glob("*.pt"))
        
        if not pt_files:
            logger.warning("[PRMDataStorage] No .pt files found in output directory")
            return {
                "output_dir": str(self.output_dir),
                "num_files": 0,
                "total_size_mb": 0.0,
            }
        
        # Compute statistics
        total_size = sum(f.stat().st_size for f in pt_files)
        
        stats = {
            "output_dir": str(self.output_dir),
            "num_files": len(pt_files),
            "total_size_mb": total_size / (1024 * 1024),
            "files": [
                {
                    "name": f.name,
                    "size_kb": f.stat().st_size / 1024,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                }
                for f in sorted(pt_files, key=lambda x: x.stat().st_mtime, reverse=True)
            ]
        }
        
        logger.info(f"[PRMDataStorage] Found {len(pt_files)} files, "
                   f"total size: {stats['total_size_mb']:.2f} MB")
        
        return stats
    
    def create_dataset_index(self, output_filename: str = "dataset_index.json") -> str:
        """Create an index file for all saved data.
        
        Args:
            output_filename: Name of the index file
            
        Returns:
            Path to the index file
        """
        logger.info("[PRMDataStorage] Creating dataset index")
        
        # Find all .pt files
        pt_files = list(self.output_dir.glob("*.pt"))
        
        index = {
            "created_at": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "num_files": len(pt_files),
            "files": []
        }
        
        # Process each file
        for pt_file in sorted(pt_files):
            try:
                # Load file to get metadata
                data = torch.load(pt_file)
                
                file_info = {
                    "filename": pt_file.name,
                    "filepath": str(pt_file),
                    "size_kb": pt_file.stat().st_size / 1024,
                }
                
                # Add type-specific information
                if "batch_name" in data:
                    # Batch file
                    file_info.update({
                        "type": "batch",
                        "batch_name": data["batch_name"],
                        "num_questions": data["num_questions"],
                    })
                elif "question_id" in data:
                    # Single question file
                    file_info.update({
                        "type": "question",
                        "question_id": data["question_id"],
                        "is_correct": data.get("is_correct"),
                        "num_paths": len(data.get("paths", [])),
                    })
                
                index["files"].append(file_info)
                
            except Exception as e:
                logger.warning(f"[PRMDataStorage] Failed to process {pt_file.name}: {e}")
                continue
        
        # Save index
        index_path = self.output_dir / output_filename
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
            logger.info(f"[PRMDataStorage] Created dataset index: {index_path}")
            logger.info(f"[PRMDataStorage] Indexed {len(index['files'])} files")
            return str(index_path)
        except Exception as e:
            logger.error(f"[PRMDataStorage] Failed to create index: {e}", exc_info=True)
            raise

