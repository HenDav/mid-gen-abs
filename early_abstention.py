import torch
import wandb
from tqdm import tqdm
from transformers import StoppingCriteria


def evaluate_abstention(model, dataset, tokenizer, thresholds=[0.3, 0.5, 0.7], use_wandb=True):
    """
    Evaluate abstention performance at different thresholds with wandb logging.

    Args:
        model: ValueHeadModel instance
        dataset: MathReasoningDataset instance
        tokenizer: HuggingFace tokenizer
        thresholds (list): List of abstention thresholds to evaluate
        use_wandb (bool): Whether to log results to wandb

    Returns:
        dict: Evaluation results
    """
    model.eval()
    results = {}

    # Track overall statistics
    all_values = []
    all_correctness = []

    print("Evaluating abstention performance...")

    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        total_samples = 0
        abstained_samples = 0
        correct_predictions = 0
        threshold_values = []
        threshold_correctness = []

        for i in tqdm(range(len(dataset)), desc=f"Threshold {threshold}", leave=False):
            item = dataset[i]
            input_ids = item['input_ids'].unsqueeze(0).to(model.device)
            true_correctness = item['correctness'].item()

            with torch.no_grad():
                generated, final_value = model.generate_with_abstention(
                    input_ids, threshold=threshold, max_length=50, tokenizer=tokenizer
                )

            total_samples += 1

            if final_value is not None:
                threshold_values.append(final_value)
                threshold_correctness.append(true_correctness)
                all_values.append(final_value)
                all_correctness.append(true_correctness)

            if final_value is not None and final_value < threshold:
                abstained_samples += 1
            else:
                # Predict correctness based on final value
                predicted_correctness = final_value > 0.5 if final_value is not None else False
                if predicted_correctness == bool(true_correctness):
                    correct_predictions += 1

        # Calculate metrics
        coverage = (total_samples - abstained_samples) / total_samples
        accuracy = correct_predictions / (total_samples - abstained_samples) if (total_samples - abstained_samples) > 0 else 0
        abstention_rate = abstained_samples / total_samples

        # Calculate value statistics for this threshold
        if threshold_values:
            avg_value = sum(threshold_values) / len(threshold_values)
            value_std = torch.tensor(threshold_values).std().item()

            # Calculate correlation between values and correctness
            if len(set(threshold_correctness)) > 1:
                values_tensor = torch.tensor(threshold_values)
                correctness_tensor = torch.tensor(threshold_correctness)
                correlation = torch.corrcoef(torch.stack([values_tensor, correctness_tensor]))[0, 1].item()
            else:
                correlation = 0.0
        else:
            avg_value = value_std = correlation = 0.0

        results[threshold] = {
            'coverage': coverage,
            'accuracy': accuracy,
            'abstention_rate': abstention_rate,
            'avg_value': avg_value,
            'value_std': value_std,
            'value_correctness_correlation': correlation,
            'total_samples': total_samples,
            'abstained_samples': abstained_samples,
            'correct_predictions': correct_predictions
        }

        if use_wandb and wandb.run:
            wandb.log({
                f"eval/threshold_{threshold}/coverage": coverage,
                f"eval/threshold_{threshold}/accuracy": accuracy,
                f"eval/threshold_{threshold}/abstention_rate": abstention_rate,
                f"eval/threshold_{threshold}/avg_value": avg_value,
                f"eval/threshold_{threshold}/value_std": value_std,
                f"eval/threshold_{threshold}/correlation": correlation,
                f"eval/threshold_{threshold}/total_samples": total_samples,
                f"eval/threshold_{threshold}/abstained_samples": abstained_samples,
                f"eval/threshold_{threshold}/correct_predictions": correct_predictions
            })

    if use_wandb and wandb.run and all_values:
        overall_avg_value = sum(all_values) / len(all_values)
        overall_value_std = torch.tensor(all_values).std().item()

        if len(set(all_correctness)) > 1:
            overall_correlation = torch.corrcoef(
                torch.stack([torch.tensor(all_values), torch.tensor(all_correctness)])
            )[0, 1].item()
        else:
            overall_correlation = 0.0

        wandb.log({
            "eval/overall/avg_value": overall_avg_value,
            "eval/overall/value_std": overall_value_std,
            "eval/overall/correlation": overall_correlation,
            "eval/overall/total_samples": len(all_values)
        })

        wandb.log({
            "eval/value_distribution": wandb.Histogram(all_values),
            "eval/correctness_distribution": wandb.Histogram(all_correctness)
        })

    return results


class ValueStoppingCriteria(StoppingCriteria):
    """
    Efficient stopping criteria that uses hidden states from the generation process
    to calculate values without redundant forward passes through the base model.
    
    This works in conjunction with generate_with_abstention() which provides
    hidden states via output_hidden_states=True.
    """

    def __init__(self, value_model, threshold=0.5, min_tokens=10):
        """
        Initialize efficient stopping criteria.

        Args:
            value_model: ValueHeadModel instance for computing value predictions
            threshold (float): Value threshold below which to stop generation
            min_tokens (int): Minimum tokens to generate before considering stopping
        """
        self.value_model = value_model
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.last_value = None
        self.step_count = 0

    def __call__(self, input_ids, scores, **kwargs):
        """
        Check if generation should stop. The actual value computation is done
        in generate_with_abstention() using the hidden states from generation.

        Args:
            input_ids: Current generated sequence [batch_size, seq_len]
            scores: Generation scores (unused)
            **kwargs: Additional arguments that may contain hidden states

        Returns:
            bool: True if generation should stop
        """
        self.step_count += 1
        
        if input_ids.shape[1] < self.min_tokens:
            return False

        # Try to extract hidden states if available in kwargs
        hidden_states = kwargs.get('hidden_states', None)
        
        if hidden_states is not None:
            # Use the provided hidden states to compute value efficiently
            with torch.no_grad():
                # hidden_states should be from the last layer
                if isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
                    last_layer_hidden = hidden_states[-1]  # Last layer
                    last_token_hidden = last_layer_hidden[:, -1:, :]  # [batch, 1, hidden_dim]
                    
                    # Apply value head directly to the hidden state
                    value_logits = self.value_model.value_head(last_token_hidden)
                    self.last_value = torch.sigmoid(value_logits).squeeze().item()
                else:
                    # Fallback: use the hidden states directly if they're already the right shape
                    last_token_hidden = hidden_states[:, -1:, :]
                    value_logits = self.value_model.value_head(last_token_hidden)
                    self.last_value = torch.sigmoid(value_logits).squeeze().item()
        else:
            # Fallback: compute value using full forward pass (less efficient)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                values = self.value_model(input_ids=input_ids, attention_mask=attention_mask)
                self.last_value = torch.sigmoid(values[0, -1]).item()
        
        # Stop if value is below threshold
        return self.last_value < self.threshold

    def cleanup(self):
        """Clean up any resources (placeholder for future use)."""
        pass