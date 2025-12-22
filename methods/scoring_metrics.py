"""Scoring metrics module for training-free evaluation of reasoning paths.

This module implements various metrics to evaluate the quality of reasoning paths
without requiring any training, using intrinsic model properties.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from collections import Counter
import numpy as np

# Logger setup
logger = logging.getLogger(__name__)


class BaseScorer(ABC):
    """Abstract base class for path scoring metrics.
    
    All scorers should inherit from this class and implement the score method.
    """
    
    @abstractmethod
    def score(self, *args, **kwargs) -> float:
        """Compute a score for a reasoning path.
        
        Returns:
            Score in range [0, 1] where higher is better
        """
        pass
    
    def normalize_score(self, raw_score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a raw score to [0, 1] range.
        
        Args:
            raw_score: Raw score value
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized score in [0, 1]
        """
        if max_val == min_val:
            return 0.5
        normalized = (raw_score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))


class SelfConsistencyScorer(BaseScorer):
    """Scores paths based on self-consistency of generated answers.
    
    Generates multiple answers from the same path and measures agreement.
    Higher consistency indicates more reliable reasoning.
    
    Attributes:
        model_wrapper: Model wrapper for generating answers
        num_samples: Number of samples to generate for consistency check
        temperature: Temperature for sampling
    """
    
    def __init__(
        self,
        model_wrapper: Any,
        num_samples: int = 5,
        temperature: float = 0.7
    ):
        """Initialize the self-consistency scorer.
        
        Args:
            model_wrapper: Model wrapper with generation capabilities
            num_samples: Number of samples to generate
            temperature: Sampling temperature
        """
        self.model_wrapper = model_wrapper
        self.num_samples = num_samples
        self.temperature = temperature
        logger.debug(f"[SelfConsistencyScorer] Initialized with num_samples={num_samples}, temp={temperature}")
    
    def score(
        self,
        path_state: Any,
        question: str = "",
        answer_extractor: Optional[callable] = None,
        **kwargs
    ) -> float:
        """Compute self-consistency score for a path.
        
        Args:
            path_state: PathState object containing the reasoning path
            question: Original question
            answer_extractor: Function to extract answer from generated text
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Consistency score in [0, 1]
        """
        logger.debug(f"[SelfConsistencyScorer] Computing score for path {path_state.path_id}")
        
        try:
            # Generate multiple answers from the same path
            answers = []
            
            for sample_idx in range(self.num_samples):
                try:
                    # Generate answer using the path's KV cache
                    answer_text = self._generate_answer_from_path(
                        path_state,
                        temperature=self.temperature
                    )
                    
                    # Extract and normalize answer
                    if answer_extractor:
                        extracted = answer_extractor(answer_text)
                    else:
                        extracted = self._default_extract_answer(answer_text)
                    
                    if extracted:
                        normalized = self._normalize_answer(extracted)
                        answers.append(normalized)
                        logger.debug(f"[SelfConsistencyScorer] Sample {sample_idx}: {normalized}")
                    else:
                        logger.debug(f"[SelfConsistencyScorer] Sample {sample_idx}: failed to extract answer")
                
                except Exception as e:
                    logger.warning(f"[SelfConsistencyScorer] Error generating sample {sample_idx}: {e}")
                    continue
            
            # Handle edge cases
            if not answers:
                logger.warning(f"[SelfConsistencyScorer] No valid answers generated for path {path_state.path_id}")
                return 0.0
            
            if len(answers) == 1:
                logger.debug(f"[SelfConsistencyScorer] Only 1 answer generated, returning default score 0.5")
                return 0.5
            
            # Calculate consistency: frequency of most common answer
            answer_counts = Counter(answers)
            most_common_answer, most_common_count = answer_counts.most_common(1)[0]
            consistency_score = most_common_count / len(answers)
            
            logger.info(f"[SelfConsistencyScorer] Path {path_state.path_id}: "
                       f"{most_common_count}/{len(answers)} agree on '{most_common_answer}' "
                       f"(score={consistency_score:.4f})")
            
            return consistency_score
        
        except Exception as e:
            logger.error(f"[SelfConsistencyScorer] Error computing score for path {path_state.path_id}: {e}")
            return 0.0
    
    def _generate_answer_from_path(
        self,
        path_state: Any,
        temperature: float
    ) -> str:
        """Generate an answer from a path using the model.
        
        Args:
            path_state: PathState object
            temperature: Sampling temperature
            
        Returns:
            Generated answer text
        """
        # This is a simplified implementation
        # In practice, this would use the model_wrapper's generation capabilities
        # with the path's KV cache
        
        # For now, we need to check if model_wrapper has the necessary methods
        if not hasattr(self.model_wrapper, 'generate_from_kv_cache'):
            # Fallback: return empty string if method not available
            logger.debug("[SelfConsistencyScorer] model_wrapper does not have generate_from_kv_cache method")
            return ""
        
        return self.model_wrapper.generate_from_kv_cache(
            kv_cache=path_state.kv_cache,
            max_new_tokens=256,
            temperature=temperature
        )
    
    def _default_extract_answer(self, text: str) -> Optional[str]:
        """Default answer extraction logic.
        
        Args:
            text: Generated text
            
        Returns:
            Extracted answer or None
        """
        # Try to extract from \boxed{} for math problems
        import re
        boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxes:
            return boxes[-1].strip()
        
        # Try to extract last number
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]
        
        # Return last non-empty line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return None
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.
        
        Args:
            answer: Raw answer string
            
        Returns:
            Normalized answer
        """
        return answer.strip().lower()


class PerplexityScorer(BaseScorer):
    """Scores paths based on perplexity of the reasoning chain.
    
    Lower perplexity indicates the model is more confident about the reasoning.
    
    Attributes:
        model: Model for computing log probabilities
    """
    
    def __init__(self, model: Any):
        """Initialize the perplexity scorer.
        
        Args:
            model: Model with log probability computation capabilities
        """
        self.model = model
        logger.debug("[PerplexityScorer] Initialized")
    
    def score(
        self,
        path_state: Any,
        normalize: bool = True,
        **kwargs
    ) -> float:
        """Compute perplexity-based score for a path.
        
        Args:
            path_state: PathState object containing the reasoning path
            normalize: Whether to normalize by path length
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Perplexity-based score in [0, 1] (higher is better)
        """
        logger.debug(f"[PerplexityScorer] Computing score for path {path_state.path_id}")
        
        if not path_state.latent_history:
            logger.warning(f"[PerplexityScorer] Empty latent history for path {path_state.path_id}")
            return 0.0
        
        try:
            # Try to compute perplexity from log probabilities if available
            if hasattr(path_state, 'log_probs') and path_state.log_probs:
                perplexity = self._compute_perplexity_from_logprobs(
                    path_state.log_probs,
                    normalize=normalize
                )
                logger.debug(f"[PerplexityScorer] Computed perplexity from log_probs: {perplexity:.4f}")
            else:
                # Fallback: compute from hidden states using model
                perplexity = self._compute_perplexity_from_hidden_states(
                    path_state.latent_history,
                    normalize=normalize
                )
                logger.debug(f"[PerplexityScorer] Computed perplexity from hidden states: {perplexity:.4f}")
            
            # Convert perplexity to score [0, 1] where lower perplexity = higher score
            # Typical perplexity ranges: 1-100 for good models
            # Use exponential decay to map perplexity to score
            score = self._perplexity_to_score(perplexity)
            
            logger.info(f"[PerplexityScorer] Path {path_state.path_id}: "
                       f"perplexity={perplexity:.4f}, score={score:.4f}")
            
            return score
        
        except Exception as e:
            logger.error(f"[PerplexityScorer] Error computing score for path {path_state.path_id}: {e}")
            return 0.0
    
    def _compute_perplexity_from_logprobs(
        self,
        log_probs: List[float],
        normalize: bool = True
    ) -> float:
        """Compute perplexity from log probabilities.
        
        Args:
            log_probs: List of log probabilities for each step
            normalize: Whether to normalize by length
            
        Returns:
            Perplexity value
        """
        if not log_probs:
            return float('inf')
        
        # Perplexity = exp(-mean(log_probs))
        # Handle numerical stability
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)
        
        # Clamp to avoid extreme values
        log_probs_tensor = torch.clamp(log_probs_tensor, min=-100.0, max=0.0)
        
        if normalize:
            avg_log_prob = log_probs_tensor.mean().item()
        else:
            avg_log_prob = log_probs_tensor.sum().item() / len(log_probs)
        
        perplexity = np.exp(-avg_log_prob)
        
        logger.debug(f"[PerplexityScorer] avg_log_prob={avg_log_prob:.4f}, perplexity={perplexity:.4f}")
        
        return perplexity
    
    def _compute_perplexity_from_hidden_states(
        self,
        latent_history: List[torch.Tensor],
        normalize: bool = True
    ) -> float:
        """Compute perplexity proxy from hidden states.
        
        Since we may not have direct access to log probabilities during latent generation,
        we use hidden state properties as a proxy for perplexity.
        
        Args:
            latent_history: List of latent vectors
            normalize: Whether to normalize by length
            
        Returns:
            Perplexity proxy value
        """
        if not latent_history:
            return float('inf')
        
        try:
            # Use hidden state norms and entropy as perplexity proxy
            # More stable norms and higher consistency suggest lower perplexity
            
            # 1. Compute norm stability
            norms = [vec.norm().item() for vec in latent_history]
            norm_mean = np.mean(norms)
            norm_std = np.std(norms)
            
            # 2. Compute cosine similarity between consecutive steps
            similarities = []
            for i in range(len(latent_history) - 1):
                vec1 = latent_history[i].flatten()
                vec2 = latent_history[i + 1].flatten()
                sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.5
            
            # 3. Combine metrics into perplexity proxy
            # Higher norm variance and lower similarity suggest higher perplexity
            if norm_mean > 0:
                cv = norm_std / norm_mean  # Coefficient of variation
            else:
                cv = 1.0
            
            # Perplexity proxy: higher CV and lower similarity -> higher perplexity
            # Map to reasonable perplexity range (1-100)
            perplexity_proxy = 10.0 * (1.0 + cv) / (avg_similarity + 0.1)
            
            # Clamp to reasonable range
            perplexity_proxy = max(1.0, min(100.0, perplexity_proxy))
            
            logger.debug(f"[PerplexityScorer] norm_cv={cv:.4f}, avg_sim={avg_similarity:.4f}, "
                        f"perplexity_proxy={perplexity_proxy:.4f}")
            
            return perplexity_proxy
        
        except Exception as e:
            logger.error(f"[PerplexityScorer] Error computing perplexity from hidden states: {e}")
            return 50.0  # Return middle value on error
    
    def _perplexity_to_score(self, perplexity: float) -> float:
        """Convert perplexity to score in [0, 1].
        
        Args:
            perplexity: Perplexity value (typically 1-100)
            
        Returns:
            Score in [0, 1] where lower perplexity = higher score
        """
        # Use exponential decay: score = exp(-perplexity / scale)
        # Scale chosen so perplexity=10 gives score~0.6, perplexity=50 gives score~0.1
        scale = 15.0
        score = np.exp(-perplexity / scale)
        
        # Ensure in [0, 1] range
        score = max(0.0, min(1.0, score))
        
        return score


class VerificationScorer(BaseScorer):
    """Scores paths by using the model to verify reasoning correctness.
    
    Uses prompting to ask the model to evaluate its own reasoning.
    
    Attributes:
        model_wrapper: Model wrapper for verification
        task_type: Type of task (math, code, qa, etc.)
    """
    
    def __init__(
        self,
        model_wrapper: Any,
        task_type: str = 'general'
    ):
        """Initialize the verification scorer.
        
        Args:
            model_wrapper: Model wrapper with generation capabilities
            task_type: Type of task for task-specific verification
        """
        self.model_wrapper = model_wrapper
        self.task_type = task_type
        logger.debug(f"[VerificationScorer] Initialized with task_type={task_type}")
    
    def score(
        self,
        path_state: Any,
        question: str = "",
        answer: str = "",
        **kwargs
    ) -> float:
        """Compute verification score for a path.
        
        Args:
            path_state: PathState object containing the reasoning path
            question: Original question
            answer: Generated answer
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Verification score in [0, 1]
        """
        logger.debug(f"[VerificationScorer] Computing score for path {path_state.path_id}")
        
        # If no answer provided, return neutral score
        if not answer:
            logger.debug(f"[VerificationScorer] No answer provided, returning neutral score")
            return 0.5
        
        try:
            # Create verification prompt based on task type
            verification_prompt = self._create_verification_prompt(
                question=question,
                answer=answer,
                task_type=self.task_type
            )
            
            logger.debug(f"[VerificationScorer] Verification prompt created for task_type={self.task_type}")
            
            # Generate verification response from model
            verification_response = self._generate_verification(verification_prompt)
            
            # Parse confidence from response
            confidence_score = self._parse_confidence(verification_response)
            
            logger.info(f"[VerificationScorer] Path {path_state.path_id}: "
                       f"confidence={confidence_score:.4f}")
            logger.debug(f"[VerificationScorer] Verification response: {verification_response[:100]}...")
            
            return confidence_score
        
        except Exception as e:
            logger.error(f"[VerificationScorer] Error computing score for path {path_state.path_id}: {e}")
            return 0.5  # Return neutral score on error
    
    def _create_verification_prompt(
        self,
        question: str,
        answer: str,
        task_type: str
    ) -> str:
        """Create a verification prompt based on task type.
        
        Args:
            question: Original question
            answer: Generated answer
            task_type: Type of task (math, code, qa, general)
            
        Returns:
            Verification prompt string
        """
        if task_type == 'math':
            prompt = f"""Question: {question}

Proposed Answer: {answer}

Please verify if this answer is mathematically correct. Consider:
1. Are the calculations accurate?
2. Does the answer make logical sense?
3. Are units and formatting correct?

Rate your confidence in this answer:
- High: The answer is definitely correct
- Medium: The answer seems reasonable but may have minor issues
- Low: The answer has significant problems or is likely incorrect

Confidence: """
        
        elif task_type == 'code':
            prompt = f"""Question: {question}

Proposed Code Solution:
{answer}

Please verify if this code solution is correct. Consider:
1. Does it solve the problem correctly?
2. Is the logic sound?
3. Are there any syntax or runtime errors?
4. Does it handle edge cases?

Rate your confidence in this solution:
- High: The solution is correct and well-implemented
- Medium: The solution seems reasonable but may have issues
- Low: The solution has significant problems

Confidence: """
        
        elif task_type == 'qa':
            prompt = f"""Question: {question}

Proposed Answer: {answer}

Please verify if this answer correctly addresses the question. Consider:
1. Is the answer factually accurate?
2. Does it fully address the question?
3. Is the reasoning sound?

Rate your confidence in this answer:
- High: The answer is accurate and complete
- Medium: The answer is partially correct or incomplete
- Low: The answer is incorrect or off-topic

Confidence: """
        
        else:  # general
            prompt = f"""Question: {question}

Proposed Answer: {answer}

Please verify if this answer is correct and reasonable. Evaluate the quality and correctness of the reasoning.

Rate your confidence in this answer:
- High: The answer is correct and well-reasoned
- Medium: The answer is acceptable but has some issues
- Low: The answer has significant problems

Confidence: """
        
        return prompt
    
    def _generate_verification(self, prompt: str) -> str:
        """Generate verification response from the model.
        
        Args:
            prompt: Verification prompt
            
        Returns:
            Model's verification response
        """
        try:
            # Check if model_wrapper has generation capabilities
            if hasattr(self.model_wrapper, 'generate_text'):
                response = self.model_wrapper.generate_text(
                    prompt,
                    max_new_tokens=128,
                    temperature=0.3  # Low temperature for more deterministic verification
                )
            elif hasattr(self.model_wrapper, 'vllm_generate_text_batch'):
                response = self.model_wrapper.vllm_generate_text_batch(
                    [prompt],
                    max_new_tokens=128,
                    temperature=0.3
                )[0]
            else:
                # Fallback: return neutral response
                logger.warning("[VerificationScorer] model_wrapper has no generation method")
                return "Medium"
            
            return response
        
        except Exception as e:
            logger.error(f"[VerificationScorer] Error generating verification: {e}")
            return "Medium"
    
    def _parse_confidence(self, response: str) -> float:
        """Parse confidence level from verification response.
        
        Args:
            response: Model's verification response
            
        Returns:
            Confidence score in [0, 1]
        """
        response_lower = response.lower()
        
        # Check for explicit confidence markers
        if 'high' in response_lower or 'definitely' in response_lower or 'correct' in response_lower:
            # Look for negative qualifiers
            if 'not high' in response_lower or 'not definitely' in response_lower:
                return 0.3
            return 1.0
        
        elif 'medium' in response_lower or 'reasonable' in response_lower or 'acceptable' in response_lower:
            return 0.6
        
        elif 'low' in response_lower or 'incorrect' in response_lower or 'wrong' in response_lower:
            return 0.3
        
        # Try to extract numerical confidence if present
        import re
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', response)
        if numbers:
            try:
                percentage = float(numbers[0])
                return percentage / 100.0
            except ValueError:
                pass
        
        # Default to medium confidence if unclear
        logger.debug(f"[VerificationScorer] Could not parse clear confidence, defaulting to 0.5")
        return 0.5


class HiddenStateQualityScorer(BaseScorer):
    """Scores paths based on hidden state quality metrics.
    
    Analyzes properties of hidden states like norm stability, smoothness, etc.
    
    Attributes:
        check_norm_stability: Whether to check norm stability
        check_smoothness: Whether to check smoothness
        check_entropy: Whether to check entropy
    """
    
    def __init__(
        self,
        check_norm_stability: bool = True,
        check_smoothness: bool = True,
        check_entropy: bool = False
    ):
        """Initialize the hidden state quality scorer.
        
        Args:
            check_norm_stability: Enable norm stability check
            check_smoothness: Enable smoothness check
            check_entropy: Enable entropy check
        """
        self.check_norm_stability = check_norm_stability
        self.check_smoothness = check_smoothness
        self.check_entropy = check_entropy
        logger.debug(f"[HiddenStateQualityScorer] Initialized with "
                    f"norm_stability={check_norm_stability}, "
                    f"smoothness={check_smoothness}, "
                    f"entropy={check_entropy}")
    
    def score(self, path_state: Any, **kwargs) -> float:
        """Compute hidden state quality score for a path.
        
        Args:
            path_state: PathState object containing the reasoning path
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Quality score in [0, 1]
        """
        logger.debug(f"[HiddenStateQualityScorer] Computing score for path {path_state.path_id}")
        
        if not path_state.latent_history:
            logger.warning(f"[HiddenStateQualityScorer] Empty latent history for path {path_state.path_id}")
            return 0.0
        
        scores = []
        
        # Norm stability
        if self.check_norm_stability:
            norm_score = self._compute_norm_stability(path_state.latent_history)
            scores.append(norm_score)
            logger.debug(f"[HiddenStateQualityScorer] Norm stability score: {norm_score:.4f}")
        
        # Smoothness
        if self.check_smoothness:
            smoothness_score = self._compute_smoothness(path_state.latent_history)
            scores.append(smoothness_score)
            logger.debug(f"[HiddenStateQualityScorer] Smoothness score: {smoothness_score:.4f}")
        
        # Entropy (if enabled)
        if self.check_entropy:
            entropy_score = self._compute_entropy(path_state.latent_history)
            scores.append(entropy_score)
            logger.debug(f"[HiddenStateQualityScorer] Entropy score: {entropy_score:.4f}")
        
        # Average all enabled metrics
        if not scores:
            return 0.5
        
        final_score = sum(scores) / len(scores)
        logger.debug(f"[HiddenStateQualityScorer] Final score for path {path_state.path_id}: {final_score:.4f}")
        return final_score
    
    def _compute_norm_stability(self, latent_history: List[torch.Tensor]) -> float:
        """Compute norm stability score.
        
        Args:
            latent_history: List of latent vectors
            
        Returns:
            Norm stability score in [0, 1]
        """
        try:
            norms = [vec.norm().item() for vec in latent_history]
            if len(norms) < 2:
                return 0.5
            
            # Lower standard deviation = more stable = higher score
            std_dev = np.std(norms)
            mean_norm = np.mean(norms)
            
            if mean_norm == 0:
                return 0.0
            
            # Coefficient of variation
            cv = std_dev / mean_norm
            # Convert to score (lower CV = higher score)
            score = 1.0 / (1.0 + cv)
            return score
        except Exception as e:
            logger.error(f"[HiddenStateQualityScorer] Error computing norm stability: {e}")
            return 0.0
    
    def _compute_smoothness(self, latent_history: List[torch.Tensor]) -> float:
        """Compute smoothness score based on cosine similarity progression.
        
        Args:
            latent_history: List of latent vectors
            
        Returns:
            Smoothness score in [0, 1]
        """
        try:
            if len(latent_history) < 2:
                return 0.5
            
            similarities = []
            for i in range(len(latent_history) - 1):
                vec1 = latent_history[i].flatten()
                vec2 = latent_history[i + 1].flatten()
                sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                similarities.append(sim)
            
            if not similarities:
                return 0.5
            
            # Lower variance in similarities = smoother = higher score
            std_dev = np.std(similarities)
            score = 1.0 / (1.0 + std_dev)
            return score
        except Exception as e:
            logger.error(f"[HiddenStateQualityScorer] Error computing smoothness: {e}")
            return 0.0
    
    def _compute_entropy(self, latent_history: List[torch.Tensor]) -> float:
        """Compute entropy-based score.
        
        Args:
            latent_history: List of latent vectors
            
        Returns:
            Entropy score in [0, 1]
        """
        try:
            if len(latent_history) < 2:
                return 0.5
            
            # Compute entropy of hidden state distributions
            # We approximate this by looking at the distribution of values in the hidden states
            entropies = []
            
            for hidden_vec in latent_history:
                # Flatten the tensor
                values = hidden_vec.flatten()
                
                # Create histogram to approximate distribution
                # Use 50 bins for reasonable granularity
                hist, _ = torch.histogram(values.cpu().float(), bins=50)
                
                # Normalize to get probabilities
                probs = hist.float() / hist.sum()
                
                # Filter out zero probabilities
                probs = probs[probs > 0]
                
                # Compute entropy: H = -sum(p * log(p))
                entropy = -(probs * torch.log(probs)).sum().item()
                entropies.append(entropy)
            
            if not entropies:
                return 0.5
            
            # Average entropy across all steps
            avg_entropy = np.mean(entropies)
            
            # Normalize entropy to [0, 1]
            # Typical entropy range for 50 bins: 0 to log(50) ≈ 3.9
            max_entropy = np.log(50)
            normalized_entropy = avg_entropy / max_entropy
            
            # Higher entropy can indicate more uncertainty
            # For quality score, moderate entropy is good (not too low, not too high)
            # Use inverted U-shape: peak at 0.5 entropy
            if normalized_entropy < 0.5:
                score = 2 * normalized_entropy
            else:
                score = 2 * (1 - normalized_entropy)
            
            score = max(0.0, min(1.0, score))
            
            logger.debug(f"[HiddenStateQualityScorer] avg_entropy={avg_entropy:.4f}, "
                        f"normalized={normalized_entropy:.4f}, score={score:.4f}")
            
            return score
        
        except Exception as e:
            logger.error(f"[HiddenStateQualityScorer] Error computing entropy: {e}")
            return 0.5


class EnsembleScorer(BaseScorer):
    """Combines multiple scoring metrics with configurable weights.
    
    Attributes:
        scorers: Dictionary mapping scorer names to (scorer, weight) tuples
        weights: Dictionary of weights for each scorer
    """
    
    def __init__(
        self,
        scorers: Optional[Dict[str, Tuple[BaseScorer, float]]] = None,
        default_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the ensemble scorer.
        
        Args:
            scorers: Dictionary of {name: (scorer, weight)} pairs
            default_weights: Default weights if not specified in scorers
        """
        self.scorers: Dict[str, Tuple[BaseScorer, float]] = scorers or {}
        self.default_weights = default_weights or {
            'self_consistency': 0.4,
            'perplexity': 0.3,
            'verification': 0.2,
            'hidden_quality': 0.1,
        }
        logger.info(f"[EnsembleScorer] Initialized with {len(self.scorers)} scorers")
        logger.debug(f"[EnsembleScorer] Default weights: {self.default_weights}")
    
    def add_scorer(self, name: str, scorer: BaseScorer, weight: float = 1.0) -> None:
        """Add a scorer to the ensemble.
        
        Args:
            name: Name for this scorer
            scorer: Scorer instance
            weight: Weight for this scorer in the ensemble
        """
        self.scorers[name] = (scorer, weight)
        logger.info(f"[EnsembleScorer] Added scorer '{name}' with weight {weight:.3f}")
    
    def remove_scorer(self, name: str) -> bool:
        """Remove a scorer from the ensemble.
        
        Args:
            name: Name of the scorer to remove
            
        Returns:
            True if scorer was removed, False if not found
        """
        if name in self.scorers:
            del self.scorers[name]
            logger.info(f"[EnsembleScorer] Removed scorer '{name}'")
            return True
        return False
    
    def score(
        self,
        path_state: Any,
        **kwargs
    ) -> float:
        """Compute ensemble score for a path.
        
        Args:
            path_state: PathState object containing the reasoning path
            **kwargs: Additional arguments passed to individual scorers
            
        Returns:
            Weighted ensemble score in [0, 1]
        """
        logger.debug(f"[EnsembleScorer] Computing ensemble score for path {path_state.path_id}")
        
        if not self.scorers:
            logger.warning("[EnsembleScorer] No scorers available, returning default score")
            return 0.5
        
        weighted_scores = []
        total_weight = 0.0
        score_breakdown = {}
        
        for name, (scorer, weight) in self.scorers.items():
            try:
                # Call scorer with appropriate arguments
                if hasattr(scorer, 'score'):
                    score = scorer.score(path_state, **kwargs)
                else:
                    logger.warning(f"[EnsembleScorer] Scorer '{name}' has no score method")
                    continue
                
                weighted_scores.append(score * weight)
                total_weight += weight
                score_breakdown[name] = score
                logger.debug(f"[EnsembleScorer] {name}: {score:.4f} (weight={weight:.3f})")
            except Exception as e:
                logger.error(f"[EnsembleScorer] Error computing score for '{name}': {e}")
                continue
        
        if total_weight == 0:
            logger.warning("[EnsembleScorer] Total weight is zero, returning default score")
            return 0.5
        
        # Compute weighted average
        ensemble_score = sum(weighted_scores) / total_weight
        
        logger.info(f"[EnsembleScorer] Ensemble score for path {path_state.path_id}: {ensemble_score:.4f}")
        logger.debug(f"[EnsembleScorer] Score breakdown: {score_breakdown}")
        
        return ensemble_score
    
    def score_with_breakdown(
        self,
        path_state: Any,
        **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """Compute ensemble score with detailed breakdown.
        
        Args:
            path_state: PathState object containing the reasoning path
            **kwargs: Additional arguments passed to individual scorers
            
        Returns:
            Tuple of (ensemble_score, score_breakdown_dict)
        """
        logger.debug(f"[EnsembleScorer] Computing ensemble score with breakdown for path {path_state.path_id}")
        
        if not self.scorers:
            return 0.5, {}
        
        weighted_scores = []
        total_weight = 0.0
        score_breakdown = {}
        
        for name, (scorer, weight) in self.scorers.items():
            try:
                score = scorer.score(path_state, **kwargs)
                weighted_scores.append(score * weight)
                total_weight += weight
                score_breakdown[name] = {
                    'score': score,
                    'weight': weight,
                    'weighted_score': score * weight
                }
            except Exception as e:
                logger.error(f"[EnsembleScorer] Error computing score for '{name}': {e}")
                score_breakdown[name] = {
                    'score': 0.0,
                    'weight': weight,
                    'weighted_score': 0.0,
                    'error': str(e)
                }
        
        if total_weight == 0:
            return 0.5, score_breakdown
        
        ensemble_score = sum(weighted_scores) / total_weight
        
        logger.info(f"[EnsembleScorer] Ensemble score for path {path_state.path_id}: {ensemble_score:.4f}")
        
        return ensemble_score, score_breakdown
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update weights for scorers in the ensemble.
        
        Args:
            weights: Dictionary mapping scorer names to new weights
        """
        for name, weight in weights.items():
            if name in self.scorers:
                scorer, _ = self.scorers[name]
                self.scorers[name] = (scorer, weight)
                logger.info(f"[EnsembleScorer] Updated weight for '{name}': {weight:.3f}")
            else:
                logger.warning(f"[EnsembleScorer] Scorer '{name}' not found, cannot update weight")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights for all scorers.
        
        Returns:
            Dictionary mapping scorer names to weights
        """
        return {name: weight for name, (_, weight) in self.scorers.items()}
    
    def normalize_weights(self) -> None:
        """Normalize weights so they sum to 1.0."""
        total_weight = sum(weight for _, weight in self.scorers.values())
        
        if total_weight == 0:
            logger.warning("[EnsembleScorer] Cannot normalize: total weight is zero")
            return
        
        for name, (scorer, weight) in self.scorers.items():
            normalized_weight = weight / total_weight
            self.scorers[name] = (scorer, normalized_weight)
        
        logger.info("[EnsembleScorer] Normalized weights to sum to 1.0")
    
    def __repr__(self) -> str:
        """String representation of the ensemble scorer."""
        scorer_info = ", ".join([f"{name}({w:.2f})" for name, (_, w) in self.scorers.items()])
        return f"EnsembleScorer({scorer_info})"

