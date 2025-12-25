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
        logger.debug(f"[SelfConsistencyScorer] Will generate {self.num_samples} samples "
                    f"with temperature={self.temperature:.2f}")
        
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
                logger.warning(f"[SelfConsistencyScorer] No valid answers generated for path {path_state.path_id} "
                             f"(attempted {self.num_samples} samples). Returning score 0.0")
                return 0.0
            
            if len(answers) == 1:
                logger.debug(f"[SelfConsistencyScorer] Only 1 valid answer generated out of {self.num_samples} samples "
                           f"for path {path_state.path_id}, returning default score 0.5")
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
        import torch
        
        logger.debug(f"[SelfConsistencyScorer] Generating answer from path {path_state.path_id} "
                    f"with temperature={temperature:.2f}")
        
        try:
            # Check if we have a valid kv_cache
            if path_state.kv_cache is None:
                logger.warning(f"[SelfConsistencyScorer] Path {path_state.path_id} has no kv_cache")
                return ""
            
            # Check if model_wrapper has the necessary methods
            if not hasattr(self.model_wrapper, 'generate_text_batch'):
                logger.error("[SelfConsistencyScorer] model_wrapper does not have generate_text_batch method")
                return ""
            
            if not hasattr(self.model_wrapper, 'tokenizer'):
                logger.error("[SelfConsistencyScorer] model_wrapper does not have tokenizer")
                return ""
            
            # Create a continuation token to trigger generation from kv_cache
            # We use a newline token or BOS token to continue generation
            tokenizer = self.model_wrapper.tokenizer
            
            # Try to use a neutral continuation token (newline or space)
            continuation_text = "\n"
            continuation_ids = tokenizer.encode(continuation_text, add_special_tokens=False, return_tensors="pt")
            
            # If tokenization failed, use a single padding token
            if continuation_ids.shape[1] == 0:
                continuation_ids = torch.tensor([[tokenizer.pad_token_id or 0]], dtype=torch.long)
            
            continuation_ids = continuation_ids.to(self.model_wrapper.device)
            
            # Create attention mask for the continuation token
            attention_mask = torch.ones_like(continuation_ids, dtype=torch.long)
            
            # Generate text using the kv_cache from this path
            logger.debug(f"[SelfConsistencyScorer] Calling generate_text_batch with kv_cache "
                        f"from path {path_state.path_id}")
            
            generated_texts, _ = self.model_wrapper.generate_text_batch(
                input_ids=continuation_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=temperature,
                top_p=0.95,
                past_key_values=path_state.kv_cache
            )
            
            # Extract the first (and only) generated text
            if generated_texts and len(generated_texts) > 0:
                answer_text = generated_texts[0]
                logger.debug(f"[SelfConsistencyScorer] Generated answer preview: {answer_text[:100]}...")
                return answer_text
            else:
                logger.warning(f"[SelfConsistencyScorer] No text generated for path {path_state.path_id}")
                return ""
                
        except Exception as e:
            logger.error(f"[SelfConsistencyScorer] Error generating answer from path {path_state.path_id}: {e}")
            logger.debug(f"[SelfConsistencyScorer] Exception details:", exc_info=True)
            return ""
    
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
            
            logger.debug(f"[PerplexityScorer] Path {path_state.path_id}: "
                       f"perplexity={perplexity:.4f}, score={score:.4f}")
            
            # Clean up any temporary tensors created during scoring
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
        
        # Clean up temporary tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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


class LatentConsistencyScorer(BaseScorer):
    """Scores paths based on latent-level consistency across multiple paths.
    
    Directly compares the hidden states (latents) from different paths without decoding.
    This is much faster than text-based self-consistency checking.
    
    Attributes:
        similarity_metric: Method to compute similarity ('cosine', 'euclidean', 'l2', 'kl_divergence')
        aggregation_method: How to aggregate pairwise similarities ('mean', 'min', 'max')
        use_last_latent: Whether to use only the last latent or average all latents
    """
    
    def __init__(
        self,
        similarity_metric: str = 'cosine',
        aggregation_method: str = 'mean',
        use_last_latent: bool = True
    ):
        """Initialize the latent consistency scorer.
        
        Args:
            similarity_metric: Method for computing similarity between latents
                - 'cosine': Cosine similarity (default)
                - 'euclidean': Euclidean distance
                - 'l2': L2 normalized distance
                - 'kl_divergence': KL divergence (treats vectors as probability distributions)
            aggregation_method: Method for aggregating pairwise similarities
            use_last_latent: If True, use only the last latent; if False, average all latents
        """
        self.similarity_metric = similarity_metric
        self.aggregation_method = aggregation_method
        self.use_last_latent = use_last_latent
        
        logger.info(f"[LatentConsistencyScorer] Initialized with similarity_metric={similarity_metric}, "
                   f"aggregation={aggregation_method}, use_last_latent={use_last_latent}")
        logger.debug(f"[LatentConsistencyScorer] Supported metrics: cosine, euclidean, l2, kl_divergence")

    def score(self, path_states: List[Any], **kwargs) -> float:
        """计算组级潜在一致性分数
        
        Args:
            path_states: List of path states to score
            **kwargs: Additional arguments
            
        Returns:
            Consistency score in [0, 1], higher means more consistent
        """
        logger.debug(f"[LatentConsistencyScorer] Computing group-level consistency score for {len(path_states)} paths")
        
        latent_vectors = []
        for i, path_state in enumerate(path_states):
            vec = self._extract_latent_representation(path_state)
            if vec is not None:
                latent_vectors.append(vec)
                logger.debug(f"[LatentConsistencyScorer] Extracted latent vector from path {i}, shape: {vec.shape}")

        if len(latent_vectors) < 2:
            logger.info(f"[LatentConsistencyScorer] Insufficient latent vectors ({len(latent_vectors)}), returning default score 0.5")
            return 0.5

        # 计算质心(共识表示)
        centroid = torch.stack(latent_vectors).mean(dim=0)
        logger.debug(f"[LatentConsistencyScorer] Computed centroid from {len(latent_vectors)} vectors")

        # 测量每个路径到质心的距离/相似度
        distances = []
        for i, vec in enumerate(latent_vectors):
            if self.similarity_metric == 'cosine':
                sim = F.cosine_similarity(vec.unsqueeze(0), centroid.unsqueeze(0)).item()
                dist = (1.0 - sim) / 2.0  # 转换为距离 [0, 1]
                distances.append(dist)
                logger.debug(f"[LatentConsistencyScorer] Path {i} cosine distance to centroid: {dist:.4f}")
            elif self.similarity_metric == 'kl_divergence':
                # For KL divergence, compute divergence from centroid
                eps = 1e-10
                p = F.softmax(vec, dim=0) + eps
                q = F.softmax(centroid, dim=0) + eps
                p = p / p.sum()
                q = q / q.sum()
                kl_div = torch.sum(p * torch.log(p / q)).item()
                distances.append(kl_div)
                logger.debug(f"[LatentConsistencyScorer] Path {i} KL divergence to centroid: {kl_div:.4f}")
            else:
                # Euclidean or L2
                dist = torch.norm(vec - centroid).item()
                distances.append(dist)
                logger.debug(f"[LatentConsistencyScorer] Path {i} {self.similarity_metric} distance to centroid: {dist:.4f}")

        # 距离越小,一致性越高
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        logger.debug(f"[LatentConsistencyScorer] Average distance: {avg_distance:.4f}, Max distance: {max_distance:.4f}")

        # 归一化为 [0, 1],距离小 = 分数高
        if self.similarity_metric == 'cosine':
            consistency = 1.0 - avg_distance  # avg_distance in [0, 1]
        else:
            # 使用指数衰减
            consistency = np.exp(-avg_distance / 10.0)

        consistency = float(max(0.0, min(1.0, consistency)))
        logger.info(f"[LatentConsistencyScorer] Final consistency score: {consistency:.4f} (metric: {self.similarity_metric})")
        
        # Clean up temporary tensors after group-level scoring
        del latent_vectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"[LatentConsistencyScorer] Cleaned up group-level scoring tensors")
        
        return consistency
    
    def _extract_latent_representation(self, path_state: Any) -> Optional[torch.Tensor]:
        """Extract a single latent vector representation from a path.
        
        Args:
            path_state: PathState object
            
        Returns:
            Latent vector tensor, or None if unavailable
        """
        try:
            if self.use_last_latent:
                # Use the last latent vector in the history
                if hasattr(path_state, 'latent_history') and path_state.latent_history:
                    latent = path_state.latent_history[-1]
                    logger.debug(f"[LatentConsistencyScorer] Using last latent from history, shape={latent.shape}")
                    return latent.flatten()
                
                # Fallback to hidden_states
                elif hasattr(path_state, 'hidden_states') and path_state.hidden_states is not None:
                    latent = path_state.hidden_states
                    logger.debug(f"[LatentConsistencyScorer] Using hidden_states as fallback, shape={latent.shape}")
                    return latent.flatten()
                
                else:
                    logger.warning("[LatentConsistencyScorer] No latent_history or hidden_states available")
                    return None
            
            else:
                # Average all latents in the history
                if hasattr(path_state, 'latent_history') and path_state.latent_history:
                    if len(path_state.latent_history) == 0:
                        return None
                    
                    # Stack and average all latents
                    stacked_latents = torch.stack([lat.flatten() for lat in path_state.latent_history])
                    averaged_latent = stacked_latents.mean(dim=0)
                    
                    logger.debug(f"[LatentConsistencyScorer] Averaged {len(path_state.latent_history)} latents, "
                               f"shape={averaged_latent.shape}")
                    return averaged_latent
                
                else:
                    logger.warning("[LatentConsistencyScorer] No latent_history available for averaging")
                    return None
        
        except Exception as e:
            logger.error(f"[LatentConsistencyScorer] Error extracting latent representation: {e}")
            return None
    
    def _compute_pairwise_similarities(self, latent_vectors: List[torch.Tensor]) -> List[float]:
        """Compute pairwise similarities between all latent vectors.
        
        Args:
            latent_vectors: List of latent vector tensors
            
        Returns:
            List of pairwise similarity scores
        """
        similarities = []
        n = len(latent_vectors)
        
        for i in range(n):
            for j in range(i + 1, n):
                vec1 = latent_vectors[i]
                vec2 = latent_vectors[j]
                
                # Ensure same shape
                if vec1.shape != vec2.shape:
                    logger.warning(f"[LatentConsistencyScorer] Shape mismatch between vectors "
                                 f"{i} and {j}: {vec1.shape} vs {vec2.shape}, skipping")
                    continue
                
                # Compute similarity based on selected metric
                similarity = self._compute_similarity(vec1, vec2)
                
                if similarity is not None:
                    similarities.append(similarity)
                    logger.debug(f"[LatentConsistencyScorer] Similarity between paths {i} and {j}: {similarity:.4f}")
        
        return similarities
    
    def _compute_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> Optional[float]:
        """Compute similarity between two latent vectors.
        
        Args:
            vec1: First latent vector
            vec2: Second latent vector
            
        Returns:
            Similarity score in [0, 1], or None on error
        """
        try:
            if self.similarity_metric == 'cosine':
                # Cosine similarity: [-1, 1] -> map to [0, 1]
                sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                # Map [-1, 1] to [0, 1]
                normalized_sim = (sim + 1.0) / 2.0
                logger.debug(f"[LatentConsistencyScorer] Cosine similarity: {sim:.4f}, normalized: {normalized_sim:.4f}")
                return normalized_sim
            
            elif self.similarity_metric == 'euclidean':
                # Euclidean distance -> convert to similarity
                distance = torch.norm(vec1 - vec2).item()
                # Use exponential decay to map distance to [0, 1]
                # Assume typical distance scale is around 10-100
                similarity = np.exp(-distance / 10.0)
                logger.debug(f"[LatentConsistencyScorer] Euclidean distance: {distance:.4f}, similarity: {similarity:.4f}")
                return similarity
            
            elif self.similarity_metric == 'l2':
                # L2 norm of difference, normalized
                diff_norm = torch.norm(vec1 - vec2).item()
                vec1_norm = torch.norm(vec1).item()
                vec2_norm = torch.norm(vec2).item()
                
                # Normalize by vector magnitudes
                avg_norm = (vec1_norm + vec2_norm) / 2.0
                if avg_norm == 0:
                    logger.debug(f"[LatentConsistencyScorer] L2: Zero norm vectors, returning {1.0 if diff_norm == 0 else 0.0}")
                    return 1.0 if diff_norm == 0 else 0.0
                
                normalized_diff = diff_norm / avg_norm
                # Convert to similarity: smaller diff = higher similarity
                similarity = 1.0 / (1.0 + normalized_diff)
                logger.debug(f"[LatentConsistencyScorer] L2 normalized diff: {normalized_diff:.4f}, similarity: {similarity:.4f}")
                return similarity
            
            elif self.similarity_metric == 'kl_divergence':
                # KL divergence: treat vectors as probability distributions
                # Lower KL divergence = higher similarity
                logger.debug(f"[LatentConsistencyScorer] Computing KL divergence between vectors")
                
                # Convert vectors to probability distributions using softmax
                # Add small epsilon to avoid log(0)
                eps = 1e-10
                
                # Apply softmax to convert to probability distributions
                p = F.softmax(vec1, dim=0) + eps
                q = F.softmax(vec2, dim=0) + eps
                
                # Normalize to ensure they sum to 1
                p = p / p.sum()
                q = q / q.sum()
                
                # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
                kl_div = torch.sum(p * torch.log(p / q)).item()
                
                # KL divergence is non-negative, typically in range [0, inf)
                # Convert to similarity score in [0, 1] using exponential decay
                # Typical KL divergence values can range from 0 to several units
                # Use scale factor to map to [0, 1] range
                similarity = np.exp(-kl_div)
                
                logger.debug(f"[LatentConsistencyScorer] KL divergence: {kl_div:.4f}, similarity: {similarity:.4f}")
                return similarity
            
            else:
                logger.warning(f"[LatentConsistencyScorer] Unknown similarity metric: {self.similarity_metric}")
                return None
        
        except Exception as e:
            logger.error(f"[LatentConsistencyScorer] Error computing similarity with metric '{self.similarity_metric}': {e}", exc_info=True)
            return None
    
    def _aggregate_similarities(self, similarities: List[float]) -> float:
        """Aggregate pairwise similarities into a single consistency score.
        
        Args:
            similarities: List of pairwise similarity scores
            
        Returns:
            Aggregated consistency score in [0, 1]
        """
        if not similarities:
            return 0.0
        
        if self.aggregation_method == 'mean':
            # Average similarity
            score = np.mean(similarities)
        
        elif self.aggregation_method == 'min':
            # Minimum similarity (most conservative)
            score = np.min(similarities)
        
        elif self.aggregation_method == 'max':
            # Maximum similarity (most optimistic)
            score = np.max(similarities)
        
        elif self.aggregation_method == 'median':
            # Median similarity
            score = np.median(similarities)
        
        else:
            logger.warning(f"[LatentConsistencyScorer] Unknown aggregation method: {self.aggregation_method}, "
                         f"using mean as fallback")
            score = np.mean(similarities)
        
        # Ensure in [0, 1] range
        return float(max(0.0, min(1.0, score)))

    def score_individual_paths(self, path_states: List[Any], **kwargs) -> List[float]:
        """为每个路径计算一致性分数(基于成对相似度的矩阵运算)

        Computes consistency scores for each path based on its average similarity
        to other paths. Higher scores indicate paths that are more consistent
        with the ensemble.

        Args:
            path_states: List of path states
            **kwargs: Additional arguments

        Returns:
            List of consistency scores for each path (0-1, higher = more consistent)
        """
        logger.debug(f"[LatentConsistencyScorer] Computing individual path scores for {len(path_states)} paths")

        latent_vectors = []
        valid_indices = []

        for i, path_state in enumerate(path_states):
            vec = self._extract_latent_representation(path_state)
            if vec is not None:
                latent_vectors.append(vec)
                valid_indices.append(i)

        if len(latent_vectors) < 2:
            logger.info(f"[LatentConsistencyScorer] Insufficient vectors for individual scoring")
            return [0.5] * len(path_states)

        # 将所有向量堆叠成矩阵 [N, D]
        X = torch.stack(latent_vectors).float()  # shape: [N, D]
        n_paths = X.shape[0]

        logger.debug(f"[LatentConsistencyScorer] Computing pairwise similarity matrix for {n_paths} paths")

        # 根据度量类型计算成对相似度矩阵
        if self.similarity_metric == 'cosine':
            # 余弦相似度矩阵计算
            # cosine_sim(i,j) = (x_i · x_j) / (||x_i|| * ||x_j||)
            X_norm = F.normalize(X, p=2, dim=1)  # L2 归一化
            similarity_matrix = torch.mm(X_norm, X_norm.t())  # [N, N]
            # 映射到 [0, 1]
            similarity_matrix = (similarity_matrix + 1.0) / 2.0

        elif self.similarity_metric == 'kl_divergence':
            # KL 散度矩阵计算（使用对称 KL 散度）
            eps = 1e-10

            # 将所有向量转换为概率分布
            P = F.softmax(X, dim=1) + eps  # [N, D]
            P = P / P.sum(dim=1, keepdim=True)  # 重新归一化

            # 计算对称 KL 散度：(KL(P||Q) + KL(Q||P)) / 2
            # 使用 log 空间计算以提高数值稳定性
            log_P = torch.log(P)  # [N, D]

            # 扩展维度用于广播
            P_expanded = P.unsqueeze(1)  # [N, 1, D]
            log_P_expanded = log_P.unsqueeze(1)  # [N, 1, D]
            log_Q_expanded = log_P.unsqueeze(0)  # [1, N, D]
            Q_expanded = P.unsqueeze(0)  # [1, N, D]

            # KL(P||Q) = sum(P * log(P/Q))
            kl_pq = torch.sum(P_expanded * (log_P_expanded - log_Q_expanded), dim=2)  # [N, N]
            # KL(Q||P) = sum(Q * log(Q/P))
            kl_qp = torch.sum(Q_expanded * (log_Q_expanded - log_P_expanded), dim=2)  # [N, N]

            # 对称 KL 散度
            symmetric_kl = (kl_pq + kl_qp) / 2.0

            # 转换为相似度分数
            similarity_matrix = torch.exp(-symmetric_kl)  # [N, N]

        elif self.similarity_metric == 'euclidean' or self.similarity_metric == 'l2':
            # 欧氏距离矩阵计算
            # 使用 PyTorch 的 cdist 函数（更高效且数值稳定）
            distance_matrix = torch.cdist(X, X, p=2)  # [N, N]

            # 转换为相似度分数
            similarity_matrix = 1.0 / (1.0 + distance_matrix)  # [N, N]

        else:
            logger.warning(f"[LatentConsistencyScorer] Unknown metric '{self.similarity_metric}', using default")
            return [0.5] * len(path_states)

        # 屏蔽对角线（不与自己比较）
        mask = torch.eye(n_paths, device=similarity_matrix.device, dtype=torch.bool)
        similarity_matrix_masked = similarity_matrix.clone()
        similarity_matrix_masked[mask] = 0.0

        # 计算每个路径与其他路径的平均相似度
        avg_similarities = similarity_matrix_masked.sum(dim=1) / (n_paths - 1)

        # 转换为 numpy 数组
        diversity_scores = avg_similarities.cpu().float().numpy()

        # 归一化到 [0, 1] 并增强区分度
        score_range = diversity_scores.max() - diversity_scores.min()

        if score_range > 1e-6:
            # Min-max 归一化
            diversity_scores = (diversity_scores - diversity_scores.min()) / score_range

            # 可选：应用非线性变换增强差异（取消注释以启用）
            # diversity_scores = np.power(diversity_scores, 0.5)  # 平方根拉开差距

            logger.info(f"[LatentConsistencyScorer] Score range after normalization: "
                        f"[{diversity_scores.min():.4f}, {diversity_scores.max():.4f}]")
        else:
            # 路径缺乏多样性
            logger.warning(f"[LatentConsistencyScorer] CRITICAL: Very low diversity detected! "
                           f"Raw score range: {score_range:.8f}")
            logger.warning(f"[LatentConsistencyScorer] All paths are nearly identical - "
                           f"diversity strategy may have failed")
            diversity_scores = np.ones_like(diversity_scores) * 0.5

        individual_scores_valid = diversity_scores.tolist()

        # 详细日志
        for i, score in enumerate(individual_scores_valid):
            logger.debug(f"[LatentConsistencyScorer] Path {valid_indices[i]}: "
                         f"avg_similarity={avg_similarities[i].item():.6f}, "
                         f"normalized_score={score:.4f}")

        # 构建完整分数列表
        individual_scores = []
        valid_idx = 0
        for i in range(len(path_states)):
            if i in valid_indices:
                individual_scores.append(individual_scores_valid[valid_idx])
                valid_idx += 1
            else:
                individual_scores.append(0.5)

        # 统计信息
        logger.info(f"[LatentConsistencyScorer] Path scoring complete - "
                    f"metric: {self.similarity_metric}, "
                    f"paths: {len(individual_scores_valid)}")
        logger.info(f"[LatentConsistencyScorer] Score statistics - "
                    f"min: {min(individual_scores_valid):.4f}, "
                    f"max: {max(individual_scores_valid):.4f}, "
                    f"mean: {np.mean(individual_scores_valid):.4f}, "
                    f"std: {np.std(individual_scores_valid):.4f}")

        # Explicitly delete large temporary tensors to free GPU memory
        logger.debug(f"[LatentConsistencyScorer] Cleaning up temporary tensors from scoring")
        del X, similarity_matrix, similarity_matrix_masked, avg_similarities
        
        # Also clean up latent vectors list
        del latent_vectors, diversity_scores
        
        # Force GPU memory cleanup and synchronization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"[LatentConsistencyScorer] GPU memory after cleanup: {gpu_mem_after:.2f}GB")

        return individual_scores


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
        
        logger.debug(f"[EnsembleScorer] Ensemble score for path {path_state.path_id}: {ensemble_score:.4f}")
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
        
        logger.debug(f"[EnsembleScorer] Ensemble score for path {path_state.path_id}: {ensemble_score:.4f}")
        
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

