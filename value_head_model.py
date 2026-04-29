"""
Transformer Model with Value Head for Early Abstention

This module implements a transformer model that can perform early abstention during inference
by predicting token-level confidence values using trl.AutoModelForCausalLMWithValueHead.
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import StoppingCriteria, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torchmetrics
from typing import Optional

import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig




class EpochMetrics:
    """
    Accumulate metrics across an epoch using TorchMetrics for efficiency.
    """
    def __init__(self, threshold, device, first_token_only=False):
        self.threshold = threshold
        self.device = device
        self.first_token_only = first_token_only
        
        # Initialize TorchMetrics for each position
        self.auroc_first = torchmetrics.AUROC(task="binary").to(device)
        self.auroc_middle = torchmetrics.AUROC(task="binary").to(device)
        self.auroc_last = torchmetrics.AUROC(task="binary").to(device)
        
        self.accuracy_first = torchmetrics.Accuracy(task="binary").to(device)
        self.accuracy_middle = torchmetrics.Accuracy(task="binary").to(device)
        self.accuracy_last = torchmetrics.Accuracy(task="binary").to(device)
        
        self.precision_first = torchmetrics.Precision(task="binary").to(device)
        self.precision_middle = torchmetrics.Precision(task="binary").to(device)
        self.precision_last = torchmetrics.Precision(task="binary").to(device)
        
        self.recall_first = torchmetrics.Recall(task="binary").to(device)
        self.recall_middle = torchmetrics.Recall(task="binary").to(device)
        self.recall_last = torchmetrics.Recall(task="binary").to(device)
        
        self.f1_first = torchmetrics.F1Score(task="binary").to(device)
        self.f1_middle = torchmetrics.F1Score(task="binary").to(device)
        self.f1_last = torchmetrics.F1Score(task="binary").to(device)
    
    def update(self, first_values, middle_values, last_values, correctness_labels):
        """Update metrics with batch predictions"""
        # Convert values to binary predictions using threshold
        first_preds = (first_values > self.threshold).int()
        middle_preds = (middle_values > self.threshold).int()
        last_preds = (last_values > self.threshold).int()
        
        labels = correctness_labels.int()
        
        # Update all metrics
        self.auroc_first.update(first_values, labels)
        self.auroc_middle.update(middle_values, labels)
        self.auroc_last.update(last_values, labels)
        
        self.accuracy_first.update(first_preds, labels)
        self.accuracy_middle.update(middle_preds, labels)
        self.accuracy_last.update(last_preds, labels)
        
        self.precision_first.update(first_preds, labels)
        self.precision_middle.update(middle_preds, labels)
        self.precision_last.update(last_preds, labels)
        
        self.recall_first.update(first_preds, labels)
        self.recall_middle.update(middle_preds, labels)
        self.recall_last.update(last_preds, labels)
        
        self.f1_first.update(first_preds, labels)
        self.f1_middle.update(middle_preds, labels)
        self.f1_last.update(last_preds, labels)
    
    def compute_and_reset(self):
        """Compute final metrics and reset for next epoch"""
        # Always include first-token metrics
        results = {
            'auroc_first': self.auroc_first.compute().item(),
            'accuracy_first': self.accuracy_first.compute().item(),
            'precision_first': self.precision_first.compute().item(),
            'recall_first': self.recall_first.compute().item(),
            'f1_first': self.f1_first.compute().item(),
        }
        
        # Add middle and last token metrics only if not in first-token-only mode
        if not self.first_token_only:
            results.update({
                'auroc_middle': self.auroc_middle.compute().item(),
                'auroc_last': self.auroc_last.compute().item(),
                'accuracy_middle': self.accuracy_middle.compute().item(),
                'accuracy_last': self.accuracy_last.compute().item(),
                'precision_middle': self.precision_middle.compute().item(),
                'precision_last': self.precision_last.compute().item(),
                'recall_middle': self.recall_middle.compute().item(),
                'recall_last': self.recall_last.compute().item(),
                'f1_middle': self.f1_middle.compute().item(),
                'f1_last': self.f1_last.compute().item(),
            })
        
        # Reset all metrics
        self.auroc_first.reset()
        self.auroc_middle.reset()
        self.auroc_last.reset()
        
        self.accuracy_first.reset()
        self.accuracy_middle.reset()
        self.accuracy_last.reset()
        
        self.precision_first.reset()
        self.precision_middle.reset()
        self.precision_last.reset()
        
        self.recall_first.reset()
        self.recall_middle.reset()
        self.recall_last.reset()
        
        self.f1_first.reset()
        self.f1_middle.reset()
        self.f1_last.reset()
        
        return results


class ValueHeadModel(nn.Module):
    """
    Value model operating on LLM embeddings from a Hugging Face model.
    Supports online embedding extraction.
    Allows pluggable value head architectures.
    """

    def __init__(
        self,
        model_name_or_path: str,
        value_head: nn.Module,
        freeze_base_model: bool = True,
        device: str = None,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load causal LM model for both embedding extraction and generation
        from transformers import AutoModelForCausalLM
        
        # Use bfloat16 for Phi-3 models (compatible with FlashAttention)
        if "phi" in model_name_or_path.lower():
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                output_hidden_states=True, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                use_cache=False  # Disable caching to avoid DynamicCache issues
            )
            self.base_model.to(self.device)
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, output_hidden_states=True, trust_remote_code=True)
            self.base_model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()

        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("✓ Base model frozen")

        self.value_head = value_head.to(self.device)
        
        # Ensure value head uses same dtype as base model for Phi-3
        if "phi" in model_name_or_path.lower():
            self.value_head = self.value_head.to(torch.bfloat16)

    def forward(self, input_ids=None, attention_mask=None, embeddings=None):
        """
        Compute scalar values per token or per sequence.
        Provide either input_ids (for online mode) or embeddings (offline mode).
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided for online mode")
        outputs = self.base_model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        # For causal LM models, use hidden_states[-1] instead of last_hidden_state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            x = outputs.hidden_states[-1]  # Last layer hidden states: [batch, seq_len, hidden]
        else:
            # Fallback for models that might have last_hidden_state
            x = outputs.last_hidden_state  # shape: [batch, seq_len, hidden]

        # Always pass attention mask to ensure causal attention
        try:
            values = self.value_head(x, attention_mask=attention_mask)
        except TypeError as e:
            if "attention_mask" in str(e):
                raise ValueError(
                    f"Value head {type(self.value_head).__name__} does not support attention_mask parameter. "
                    "Use a causal value head (e.g., Qwen3ValueHead) that supports proper attention masking."
                ) from e
            else:
                raise e
        return values
    
    def save_value_head(self, path):
        """Save only the value head weights"""
        torch.save(self.value_head.state_dict(), path)
        print(f"✓ Value head saved to {path}")
    
    def load_value_head(self, path):
        """Load value head weights"""
        state_dict = torch.load(path, map_location=self.device)
        self.value_head.load_state_dict(state_dict)
        print(f"✓ Value head loaded from {path}")
    
    def generate_with_abstention(self, input_ids, threshold=0.5, max_length=50, tokenizer=None):
        """
        Generate text with abstention capability using the existing base model
        and stopping criteria to save on generated tokens.
        
        Args:
            input_ids: Input token IDs [1, seq_len]
            threshold: Value threshold for abstention
            max_length: Maximum generation length
            tokenizer: Tokenizer for generation
            
        Returns:
            tuple: (generated_ids, final_value)
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for generation")
            
        # Import here to avoid circular dependencies
        from transformers import AutoModelForCausalLM, StoppingCriteriaList
        from early_abstention import ValueStoppingCriteria
        
        # Use the existing base model for generation
        generation_model = self.base_model
        
        # Create proper attention mask for input
        attention_mask = torch.ones_like(input_ids)
        
        # Create efficient stopping criteria that uses hidden states from generation
        value_stopping = ValueStoppingCriteria(
            value_model=self,
            threshold=threshold,
            min_tokens=10
        )
        stopping_criteria = StoppingCriteriaList([value_stopping])
        
        self.eval()
        
        # Use HuggingFace generation with stopping criteria to save tokens
        with torch.no_grad():
            outputs = generation_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Explicitly pass attention mask
                max_length=input_ids.shape[1] + max_length,
                stopping_criteria=stopping_criteria,  # This will stop early and save tokens!
                do_sample=False,  # Use greedy decoding
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Explicitly disable cache to avoid DynamicCache issues
                return_dict_in_generate=True  # Return dict to access hidden states
            )
        
        # Extract generated sequence and hidden states
        generated_ids = outputs.sequences
        hidden_states = outputs.hidden_states  # This contains hidden states for each generation step
        
        # Calculate final value using the last hidden state
        if hidden_states and len(hidden_states) > 0:
            # hidden_states is a tuple of tuples: (step1_layers, step2_layers, ...)
            # Each step contains hidden states from all layers
            last_step_hidden = hidden_states[-1]  # Last generation step
            last_layer_hidden = last_step_hidden[-1]  # Last layer of that step
            
            # Apply value head to the last token's hidden state
            with torch.no_grad():
                last_token_hidden = last_layer_hidden[:, -1:, :]  # [batch, 1, hidden_dim]
                value_logits = self.value_head(last_token_hidden)
                final_value = torch.sigmoid(value_logits).squeeze().item()
        else:
            # Fallback to stopping criteria value
            final_value = value_stopping.last_value
        
        # Clean up
        value_stopping.cleanup()
        
        return generated_ids, final_value


class TokenwiseValueHead(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),  # Dropout for regularization - automatically disabled in eval()
            nn.Linear(hidden_dim, 1)  # Scalar value per token
        )

    def forward(self, x, attention_mask=None):
        # TokenwiseValueHead doesn't use attention internally, but accepts the parameter
        # for compatibility with the ValueHeadModel interface
        return self.v_head(x).squeeze(-1)  # shape: [batch, seq_len]


def compute_tokenwise_value_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    correctness_labels: torch.Tensor,
    output_start_indices: torch.Tensor,
    first_token_only: bool = False,
):
    """
    Compute token-wise binary cross-entropy loss using value predictions.

    Args:
        model: ValueHeadModel instance
        input_ids: [batch, seq_len] – input token IDs
        attention_mask: [batch, seq_len] – 1 for real tokens, 0 for padding
        correctness_labels: [batch] – binary label for the whole sample
        output_start_indices: [batch] – index in each sequence where the output starts
        first_token_only: [bool] – if True, compute loss only on first output token

    Returns:
        torch.Tensor: scalar loss
    """
    # Compute values per token: [batch, seq_len]
    values = model(input_ids=input_ids, attention_mask=attention_mask)

    batch_size, seq_len = values.shape
    labels = correctness_labels.unsqueeze(1).expand_as(values).float()

    # Create mask for output tokens only
    output_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

    for i in range(batch_size):
        start = int(output_start_indices[i].item() if hasattr(output_start_indices[i], 'item') else output_start_indices[i])
        end = int(attention_mask[i].sum().item())
        if start < end and start < seq_len:
            if first_token_only:
                # Only mask the first output token
                output_mask[i, start] = True
            else:
                # Mask all output tokens (original behavior)
                output_mask[i, start:end] = True

    # Masked binary cross-entropy
    active_loss = output_mask.reshape(-1)
    if active_loss.sum() == 0:
        # Return a zero loss that maintains the computational graph
        return (values * 0).sum()

    active_values = values.reshape(-1)[active_loss]
    active_labels = labels.reshape(-1)[active_loss]

    return F.binary_cross_entropy_with_logits(active_values, active_labels)


def create_dynamic_collate_fn(tokenizer):
    """
    Create a custom collate function that uses the tokenizer's padding token.
    """
    def dynamic_collate_fn(batch):
        """
        Custom collate function to handle variable-length sequences without truncation.
        Pads sequences to the maximum length in the batch using tokenizer's pad_token_id.
        """
        # Find maximum length in the batch
        max_length = max(item['input_ids'].size(0) for item in batch)
        
        # Get padding token ID from tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Pad sequences to max length
        input_ids_list = []
        attention_mask_list = []
        correctness_list = []
        output_start_idx_list = []
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            correctness = item['correctness']
            output_start_idx = item['output_start_idx']
            
            # Calculate padding needed
            current_length = input_ids.size(0)
            padding_length = max_length - current_length
            
            if padding_length > 0:
                # Pad with tokenizer's pad_token_id
                input_ids = torch.cat([input_ids, torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=attention_mask.dtype)])
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            correctness_list.append(correctness)
            output_start_idx_list.append(output_start_idx)
        
        # Stack into tensors
        batch_input_ids = torch.stack(input_ids_list)
        batch_attention_mask = torch.stack(attention_mask_list)
        batch_correctness = torch.stack(correctness_list)
        batch_output_start_idx = torch.tensor(output_start_idx_list)
        
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'correctness': batch_correctness,
            'output_start_idx': batch_output_start_idx
        }
    
    return dynamic_collate_fn


def update_epoch_metrics_from_model(
    model: ValueHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    correctness_labels: torch.Tensor,
    output_start_indices: torch.Tensor,
    epoch_metrics: EpochMetrics,
    embeddings: Optional[torch.Tensor] = None,
):
    """
    Run model forward pass, extract values at first/middle/last positions, and update epoch metrics.
    """
    with torch.no_grad():
        values = model(input_ids=input_ids, attention_mask=attention_mask, embeddings=embeddings)  # [batch, seq_len]
    
    batch_size, seq_len = values.shape
    device = values.device

    first_values = torch.zeros(batch_size, device=device)
    middle_values = torch.zeros(batch_size, device=device)
    last_values = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        start_idx = output_start_indices[i].item()
        end_idx = attention_mask[i].sum().item()

        # Clamp in case indices are off (robustness)
        start_idx = min(start_idx, seq_len - 1)
        end_idx = min(end_idx, seq_len)

        # Get output slice
        output_slice = values[i, start_idx:end_idx]

        if output_slice.numel() == 0:
            first_values[i] = 0.0
            middle_values[i] = 0.0
            last_values[i] = 0.0
        else:
            first_values[i] = output_slice[0]
            middle_values[i] = output_slice[len(output_slice) // 2]
            last_values[i] = output_slice[-1]

    epoch_metrics.update(first_values, middle_values, last_values, correctness_labels)


def get_optimizer(optimizer_name, model_params, lr=1e-4, **kwargs):
    """
    Instantiate a PyTorch optimizer from its name.

    Args:
        optimizer_name (str): Name of the optimizer (e.g., 'Adam', 'SGD').
        model_params (iterable): Parameters of the model (e.g., model.parameters()).
        lr (float): Learning rate.
        **kwargs: Additional keyword arguments passed to the optimizer.

    Returns:
        torch.optim.Optimizer instance
    """
    try:
        optimizer_class = getattr(optim, optimizer_name)
    except AttributeError:
        raise ValueError(f"Optimizer '{optimizer_name}' is not found in torch.optim")

    return optimizer_class(model_params, lr=lr, **kwargs)


def load_checkpoint(model, optimizer, checkpoint_dir, epoch):
    """
    Load model and optimizer state from a specific epoch checkpoint.
    
    Args:
        model: ValueHeadModel instance
        optimizer: Optimizer instance
        checkpoint_dir: Directory containing checkpoints
        epoch: Epoch number to load
        
    Returns:
        dict: Metadata from the checkpoint
    """
    import json
    import os
    
    # Load value head weights
    value_head_path = f"{checkpoint_dir}/value_head_epoch_{epoch}.pth"
    if os.path.exists(value_head_path):
        model.load_value_head(value_head_path)
        print(f"✓ Loaded value head from epoch {epoch}")
    else:
        raise FileNotFoundError(f"Value head checkpoint not found: {value_head_path}")
    
    # Load optimizer state
    optimizer_path = f"{checkpoint_dir}/optimizer_epoch_{epoch}.pth"
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"✓ Loaded optimizer state from epoch {epoch}")
    else:
        print(f"⚠️ Optimizer checkpoint not found: {optimizer_path}")
    
    # Load metadata
    metadata_path = f"{checkpoint_dir}/metadata_epoch_{epoch}.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Loaded training metadata from epoch {epoch}")
        return metadata
    else:
        print(f"⚠️ Metadata not found: {metadata_path}")
        return {}


def train_value_head(
    model,
    dataset,
    num_epochs=3,
    lr=1e-4,
    batch_size=8,
    use_wandb=True,
    project_name="value-head-training",
    log_frequency=5,
    optimizer_name="AdamW",
    max_text_length=1000,
    save_path=None,
    optimizer_kwargs=None,
    save_every_epoch=True,
    checkpoint_dir="checkpoints",
    resume_from_epoch=None,
    first_token_only=False
):
    
    # Use custom collate for text datasets
    collate_fn = create_dynamic_collate_fn(dataset.tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # Import the function from datasets module
    from datasets import calculate_dataset_threshold
    dataset_threshold = calculate_dataset_threshold(dataset)

    if use_wandb and not wandb.run:
        wandb.init(
            project=project_name,
            config={
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "dataset_size": len(dataset),
                "model_type": "ValueHeadModel",
                "optimizer": optimizer_name,
                "log_frequency": log_frequency,
                "max_text_length": max_text_length,
                "dataset_threshold": dataset_threshold,
                "first_token_only": first_token_only
            }
        )

    # Initialize optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = get_optimizer(optimizer_name, model.value_head.parameters(), lr=lr, **optimizer_kwargs)
    
    # Create checkpoint directory if saving every epoch
    if save_every_epoch:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"✓ Checkpoint directory created: {checkpoint_dir}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if resume_from_epoch is not None:
        try:
            metadata = load_checkpoint(model, optimizer, checkpoint_dir, resume_from_epoch)
            start_epoch = resume_from_epoch
            global_step = metadata.get('global_step', 0)
            print(f"✓ Resuming training from epoch {resume_from_epoch}")
            print(f"  Previous loss: {metadata.get('avg_epoch_loss', 'N/A')}")
            print(f"  Global step: {global_step}")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
            global_step = 0
    
    model.train()

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Epochs"):
        epoch_loss = 0
        batch_losses = []
        epoch_metrics = EpochMetrics(dataset_threshold, model.device, first_token_only)

        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(epoch_pbar):
            # Handle text dataset 
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            correctness = batch["correctness"].to(model.device)
            output_start_idx = batch["output_start_idx"].to(model.device)

            optimizer.zero_grad()

            # Compute loss
            loss = compute_tokenwise_value_loss(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                correctness_labels=correctness,
                output_start_indices=output_start_idx,
                first_token_only=first_token_only
            )

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss
            global_step += 1

            # Update metrics
            update_epoch_metrics_from_model(
                model,
                input_ids,
                attention_mask,
                correctness,
                output_start_idx,
                epoch_metrics
            )

            if use_wandb and wandb.run:
                log_dict = {
                    "batch_loss": batch_loss,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "learning_rate": lr
                }

                if batch_idx % log_frequency == 0 and len(input_ids) > 0:
                    try:
                        input_sample = input_ids[0]
                        decoded_text = dataset.tokenizer.decode(input_sample, skip_special_tokens=True)
                        log_dict["sample_text"] = decoded_text[:max_text_length] + ("..." if len(decoded_text) > max_text_length else "")
                    except:
                        pass

                wandb.log(log_dict, step=global_step)

            epoch_pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}"
            })

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_std = np.std(batch_losses)
        epoch_results = epoch_metrics.compute_and_reset()

        if use_wandb and wandb.run:
            epoch_log_dict = {
                "epoch": epoch + 1,
                "epoch_loss": avg_epoch_loss,
                "epoch_loss_std": loss_std,
                "batches_per_epoch": len(dataloader),
                **epoch_results
            }
            wandb.log(epoch_log_dict, step=global_step)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} (±{loss_std:.4f})")
        if first_token_only:
            print(f"  First Token - AUROC: {epoch_results.get('auroc_first', 0):.3f}, "
                  f"F1: {epoch_results.get('f1_first', 0):.3f}, "
                  f"Accuracy: {epoch_results.get('accuracy_first', 0):.3f}")
        else:
            print(f"  AUROC - First: {epoch_results.get('auroc_first', 0):.3f}, "
                  f"Middle: {epoch_results.get('auroc_middle', 0):.3f}, "
                  f"Last: {epoch_results.get('auroc_last', 0):.3f}")
            print(f"  F1 - First: {epoch_results.get('f1_first', 0):.3f}, "
                  f"Middle: {epoch_results.get('f1_middle', 0):.3f}, "
                  f"Last: {epoch_results.get('f1_last', 0):.3f}")
        
        # Save checkpoint after each epoch
        if save_every_epoch:
            epoch_save_path = f"{checkpoint_dir}/value_head_epoch_{epoch+1}.pth"
            model.save_value_head(epoch_save_path)
            print(f"✓ Epoch {epoch+1} checkpoint saved to {epoch_save_path}")
            
            # Also save optimizer state for resuming training
            optimizer_save_path = f"{checkpoint_dir}/optimizer_epoch_{epoch+1}.pth"
            torch.save(optimizer.state_dict(), optimizer_save_path)
            
            # Save training metadata
            metadata = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'avg_epoch_loss': avg_epoch_loss,
                'loss_std': loss_std,
                'learning_rate': lr,
                'batch_size': batch_size,
                'dataset_size': len(dataset),
                'epoch_results': epoch_results
            }
            metadata_save_path = f"{checkpoint_dir}/metadata_epoch_{epoch+1}.json"
            import json
            with open(metadata_save_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log checkpoint info to wandb
            if use_wandb and wandb.run:
                wandb.log({
                    "checkpoint/epoch": epoch + 1,
                    "checkpoint/path": epoch_save_path,
                    "checkpoint/loss": avg_epoch_loss
                }, step=global_step)

    # Save final model if save_path is provided
    if save_path:
        model.save_value_head(save_path)
        print(f"✓ Final model saved to {save_path}")

    if use_wandb and wandb.run:
        try:
            artifact = wandb.Artifact(f"value-head-{wandb.run.id}", type="model")
            temp_path = f"value_head_{wandb.run.id}.pth"
            model.save_value_head(temp_path)
            artifact.add_file(temp_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"⚠️ Failed to save artifact: {e}")
