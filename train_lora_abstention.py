#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Abstention Token Prediction

This script demonstrates how to fine-tune a language model using LoRA to predict 
abstention decisions by adding new tokens "abstain" and "don't abstain" to the 
vocabulary and training the model to output these tokens followed by the original response.
"""

import torch
import sys
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import numpy as np
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import AbstractionTokenDataset, AbstractionTokenDataCollator
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    if seed is not None:
        print(f"Setting global seed to {seed} for reproducibility")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)


def compute_abstention_loss(model, batch, tokenizer, model_name):
    """Compute multiclass language modeling loss + binary classification loss for monitoring"""
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    target_ids = batch['target_ids']
    target_attention_mask = batch['target_attention_mask']
    abstention_token_ids = batch['abstention_token_ids']
    
    batch_size = input_ids.size(0)
    input_length = input_ids.size(1)
    target_length = target_ids.size(1)
    
    # Get abstention token IDs for binary classification monitoring
    from datasets import get_model_type, ABSTENTION_TOKENS
    
    model_type = get_model_type(model_name)
    token_config = ABSTENTION_TOKENS[model_type]
    
    abstain_token_id = tokenizer.convert_tokens_to_ids(token_config["abstain"])
    dont_abstain_token_id = tokenizer.convert_tokens_to_ids(token_config["dont_abstain"])
    
    # Create full sequence: input + target
    full_input_ids = torch.cat([input_ids, target_ids], dim=1)
    full_attention_mask = torch.cat([attention_mask, target_attention_mask], dim=1)
    
    # Get logits for the full sequence
    with torch.amp.autocast('cuda'):
        # Disable cache for Phi-3 compatibility
        use_cache = False if "phi" in model_name.lower() else None
        outputs = model(input_ids=full_input_ids, attention_mask=full_attention_mask, use_cache=use_cache)
        logits = outputs.logits
        
        # 1. MAIN LOSS: Standard language modeling loss on target tokens
        shift_logits = logits[:, input_length-1:-1, :].contiguous()  # Predict target tokens
        shift_labels = target_ids.contiguous()  # Target tokens to predict
        
        # Flatten for loss computation
        lm_loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Only compute loss on active (non-padded) target positions
        active_loss = target_attention_mask.reshape(-1) == 1
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
        active_labels = shift_labels.view(-1)[active_loss]
        
        if len(active_labels) == 0:
            # No active labels, return zero loss
            main_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            classification_loss_val = 0.0
        else:
            main_loss = lm_loss_fct(active_logits, active_labels)
            
            # 2. MONITORING: Binary classification loss on first target token
            first_target_logits = logits[:, input_length-1, :]  # Shape: [batch_size, vocab_size]
            
            # Extract logits for abstention classification
            abstain_logits = first_target_logits[:, abstain_token_id]
            dont_abstain_logits = first_target_logits[:, dont_abstain_token_id]
            
            # Create binary classification logits
            binary_logits = torch.stack([dont_abstain_logits, abstain_logits], dim=1)  # [batch_size, 2]
            
            # Create binary labels (0 = don't abstain, 1 = abstain)
            binary_labels = (abstention_token_ids == abstain_token_id).long()
            
            # Compute binary classification loss (for monitoring only)
            binary_loss_fct = nn.CrossEntropyLoss()
            classification_loss = binary_loss_fct(binary_logits, binary_labels)
            classification_loss_val = classification_loss.item()
    
    # Clean up intermediate tensors
    del outputs, logits, shift_logits
    if 'active_logits' in locals():
        del active_logits, active_labels
    torch.cuda.empty_cache()
    
    # Return main loss (language modeling) and binary classification loss for monitoring
    return main_loss, classification_loss_val


def save_checkpoint(model, optimizer, scaler, epoch, global_step, loss, config, checkpoint_dir="checkpoints"):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': config
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{global_step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint


def evaluate_model(model, val_loader, tokenizer, config):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_classification_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Compute loss
            loss, classification_loss_val = compute_abstention_loss(model, batch, tokenizer, config["model_name"])
            
            total_loss += loss.item()
            total_classification_loss += classification_loss_val
            num_batches += 1
            
            # Clean up
            del batch
    
    avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_val_classification_loss = total_classification_loss / num_batches if num_batches > 0 else float('inf')
    
    model.train()  # Switch back to training mode
    return avg_val_loss, avg_val_classification_loss


def train_lora_abstention(model, dataset, tokenizer, config):
    """Train the model with LoRA for abstention token prediction with validation evaluation"""
    print("Starting LoRA training...")
    
    # Ensure reproducibility with global seeding
    if config.get("data_split_seed") is not None:
        set_seed(config["data_split_seed"])
    
    # Split dataset with optional fixed seed for reproducible validation splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use fixed seed if provided for consistent validation splits across hyperparameter trials
    generator = None
    if config.get("data_split_seed") is not None:
        generator = torch.Generator().manual_seed(config["data_split_seed"])
        print(f"Using fixed seed {config['data_split_seed']} for train/validation split and global reproducibility")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    # Create data loaders
    data_collator = AbstractionTokenDataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=data_collator
    )
    
    # Initialize optimizer and scaler for mixed precision
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()
    
    # Gradient accumulation settings
    gradient_accumulation_steps = 10
    
    # Training tracking with validation metrics
    global_step = 0
    all_step_losses = []  # Track losses per gradient step
    epoch_summaries = []
    recent_step_losses = []  # Track recent step losses for rolling average
    accumulated_loss = 0.0
    start_epoch = 0
    best_val_loss = float('inf')
    val_losses = []  # Track validation losses
    val_classification_losses = []  # Track validation classification losses
    
    # Checkpointing setup
    checkpoint_dir = os.path.join(config["output_dir"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Try to resume from checkpoint if specified
    if config.get("resume_from_checkpoint"):
        checkpoint_path = config["resume_from_checkpoint"]
        if checkpoint_path == "latest":
            checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scaler)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            print(f"✓ Resuming from epoch {start_epoch}, step {global_step}")
    
    model.train()
    
    for epoch in tqdm(range(start_epoch, config["num_epochs"]), desc="Training Epochs"):
        epoch_step_losses = []  # Track step losses for this epoch
        epoch_total_loss = 0.0
        num_steps_this_epoch = 0
        
        # Training loop with corrected loss reporting
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, batch in enumerate(epoch_pbar):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Compute loss with mixed precision
            with torch.amp.autocast('cuda'):
                loss, classification_loss_val = compute_abstention_loss(model, batch, tokenizer, config["model_name"])
                # Scale loss by accumulation steps for proper averaging
                scaled_loss = loss / gradient_accumulation_steps
            
            # Scale loss and backward pass
            scaler.scale(scaled_loss).backward()
            
            # FIXED: Track raw loss (not scaled)
            accumulated_loss += loss.item()
            accumulated_classification_loss = getattr(train_lora_abstention, 'accumulated_classification_loss', 0.0)
            accumulated_classification_loss += classification_loss_val
            train_lora_abstention.accumulated_classification_loss = accumulated_classification_loss
            
            # Update weights and log metrics only after gradient accumulation step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Update weights after accumulating gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                num_steps_this_epoch += 1
                
                # FIXED: Calculate step losses correctly
                step_loss = accumulated_loss / gradient_accumulation_steps
                step_classification_loss = train_lora_abstention.accumulated_classification_loss / gradient_accumulation_steps
                
                # Track losses
                epoch_step_losses.append(step_loss)
                all_step_losses.append(step_loss)
                recent_step_losses.append(step_loss)
                epoch_total_loss += step_loss
                
                # Keep rolling window of recent losses
                rolling_window = 20
                if len(recent_step_losses) > rolling_window:
                    recent_step_losses.pop(0)
                
                # Calculate rolling average
                rolling_avg_loss = sum(recent_step_losses) / len(recent_step_losses)
                
                # Log to wandb
                if config["use_wandb"] and wandb.run:
                    wandb.log({
                        "train/step_loss": step_loss,
                        "train/classification_loss": step_classification_loss,
                        "train/rolling_avg_loss": rolling_avg_loss,
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                    })
                
                # Reset accumulated losses
                accumulated_loss = 0.0
                train_lora_abstention.accumulated_classification_loss = 0.0
                
                # FIXED: Update progress bar with both raw and classification loss
                current_epoch_avg = epoch_total_loss / num_steps_this_epoch
                epoch_pbar.set_postfix({
                    "raw": f"{step_loss:.2f}",
                    "cls": f"{step_classification_loss:.3f}",
                    "avg": f"{current_epoch_avg:.3f}",
                    "step": global_step
                })
            else:
                # For non-accumulation steps, show current batch info
                epoch_pbar.set_postfix({
                    "raw": f"{loss.item():.2f}",
                    "cls": f"{classification_loss_val:.3f}",
                    "acc": f"{(batch_idx + 1) % gradient_accumulation_steps}/{gradient_accumulation_steps}"
                })
            
            # Clear cache periodically
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
            
            # Delete batch tensors explicitly
            del batch
        
        # FIXED: Calculate epoch statistics correctly
        if epoch_step_losses:
            avg_epoch_loss = sum(epoch_step_losses) / len(epoch_step_losses)
            min_epoch_loss = min(epoch_step_losses)
            max_epoch_loss = max(epoch_step_losses)
            epoch_loss_std = np.std(epoch_step_losses)
        else:
            avg_epoch_loss = min_epoch_loss = max_epoch_loss = epoch_loss_std = 0.0
        
        # Store epoch summary (no console output during training)
        epoch_summary = {
            'epoch': epoch + 1,
            'avg_loss': avg_epoch_loss,
            'min_loss': min_epoch_loss,
            'max_loss': max_epoch_loss,
            'loss_std': epoch_loss_std,
            'num_steps': num_steps_this_epoch,
            'num_batches': len(train_loader)
        }
        epoch_summaries.append(epoch_summary)
        
        # Evaluate on validation set (no console output during training)
        val_loss, val_classification_loss = evaluate_model(model, val_loader, tokenizer, config)
        val_losses.append(val_loss)
        val_classification_losses.append(val_classification_loss)
        
        # Track best validation loss for monitoring
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Update epoch summary with validation metrics
        epoch_summary.update({
            'val_loss': val_loss,
            'val_classification_loss': val_classification_loss,
            'is_best_val': val_loss == best_val_loss
        })
        
        # Log validation metrics to wandb
        if config["use_wandb"] and wandb.run:
            wandb.log({
                "epoch/avg_step_loss": avg_epoch_loss,
                "epoch/min_step_loss": min_epoch_loss,
                "epoch/max_step_loss": max_epoch_loss,
                "epoch/loss_std": epoch_loss_std,
                "epoch/val_loss": val_loss,
                "epoch/val_classification_loss": val_classification_loss,
                "epoch/num_steps": num_steps_this_epoch,
                "epoch/num_batches": len(train_loader),
                "epoch/epoch_number": epoch + 1,
                "epoch/best_val_loss": best_val_loss
            })
        
        # Save checkpoint after each epoch if configured
        if config.get("save_every_epoch", False):
            save_checkpoint(model, optimizer, scaler, epoch + 1, global_step, avg_epoch_loss, config, checkpoint_dir)
        
        # Clear memory after each epoch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Final summary with validation results
    print(f"Training completed - {len(epoch_summaries)} epochs, {global_step} steps")
    if all_step_losses:
        overall_avg_loss = sum(all_step_losses) / len(all_step_losses)
        print(f"Overall average training loss: {overall_avg_loss:.4f}")
    
    if val_losses:
        final_val_loss = val_losses[-1]
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Print final classification validation loss for hyperparameter search parsing
    if val_classification_losses:
        final_val_classification_loss = val_classification_losses[-1]
        print(f"FINAL_VALIDATION_CLASSIFICATION_LOSS: {final_val_classification_loss:.6f}")
    
    # Log final summary to wandb
    if config["use_wandb"] and wandb.run:
        final_metrics = {
            "final/total_epochs": len(epoch_summaries),
            "final/total_steps": global_step,
            "final/best_val_loss": best_val_loss
        }
        
        if all_step_losses:
            final_metrics.update({
                "final/overall_avg_step_loss": sum(all_step_losses) / len(all_step_losses),
                "final/best_step_loss": min(all_step_losses),
                "final/worst_step_loss": max(all_step_losses)
            })
        
        if epoch_summaries:
            epoch_avg_losses = [ep['avg_loss'] for ep in epoch_summaries]
            final_metrics.update({
                "final/best_epoch_avg_loss": min(epoch_avg_losses),
                "final/worst_epoch_avg_loss": max(epoch_avg_losses)
            })
        
        if val_losses:
            final_metrics.update({
                "final/final_val_loss": val_losses[-1],
                "final/best_val_loss": best_val_loss,
                "final/worst_val_loss": max(val_losses)
            })
        
        wandb.log(final_metrics)
    
    return model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Abstention Token Prediction")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name or path of the model to fine-tune")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the training data JSONL file")
    parser.add_argument("--output_dir", type=str, default="./lora_abstention_checkpoints",
                        help="Directory to save model and checkpoints")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (None for all)")
    parser.add_argument("--data_split_seed", type=int, default=None,
                        help="Random seed for train/validation split (None for random)")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="lora-abstention-training",
                        help="W&B project name")
    parser.add_argument("--experiment_name", type=str, default="abstention_token_prediction",
                        help="W&B experiment name")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"],
                        help="W&B logging mode")
    
    # Device and checkpointing arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use for training (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (use 'latest' for latest checkpoint)")
    parser.add_argument("--save_every_epoch", action="store_true", default=True,
                        help="Save checkpoint after each epoch")
    parser.add_argument("--no_save_every_epoch", action="store_true",
                        help="Disable saving checkpoint after each epoch")
    
    args = parser.parse_args()
    
    # Handle wandb flag conflicts
    if args.no_wandb:
        args.use_wandb = False
    
    # Handle checkpoint saving flag conflicts
    if args.no_save_every_epoch:
        args.save_every_epoch = False
    
    return args


def main():
    """Main training and evaluation pipeline with wandb tracking"""
    
    print("=" * 60)
    print("LoRA Abstention Token Training with W&B Tracking")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_args()
    
    # Set global seed for reproducibility if data_split_seed is provided
    if args.data_split_seed is not None:
        set_seed(args.data_split_seed)
    
    # Convert args to config dictionary for compatibility
    config = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "data_split_seed": args.data_split_seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "use_wandb": args.use_wandb,
        "project_name": args.project_name,
        "experiment_name": args.experiment_name,
        "wandb_mode": args.wandb_mode,
        "device": args.device,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "save_every_epoch": args.save_every_epoch
    }
    
    # Initialize wandb
    if config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            name=config["experiment_name"],
            config=config,
            mode=config["wandb_mode"],
            tags=["lora", "abstention", "mathematical-reasoning", "qwen"]
        )
        print(f"✓ W&B initialized ({config['wandb_mode']} mode)")
        if config["wandb_mode"] == "online":
            print(f"  View at: {wandb.run.url}")
    
    print(f"Model: {config['model_name']}")
    print(f"Data: {config['data_path']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print()
    
    # Configure device
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    
    print(f"Using device: {device}")
    
    # Validate device availability
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        elif device.index is not None and device.index >= torch.cuda.device_count():
            print(f"⚠️ CUDA device {device.index} not available, using cuda:0")
            device = torch.device("cuda:0")
        else:
            print(f"✓ CUDA device validated: {device}")
    
    # Create output directory
    try:
        os.makedirs(config["output_dir"], exist_ok=True)
        print(f"✓ Output directory created: {config['output_dir']}")
    except Exception as e:
        print(f"✗ Error creating output directory: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        return
    
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Handle device mapping based on device configuration
        if device.type == "cuda":
            # If specific CUDA device is requested, use that device only
            if device.index is not None:
                device_map = {"": device.index}  # Map all modules to specific GPU
            else:
                device_map = "auto"  # Use automatic device mapping
        else:
            device_map = None  # CPU mode
            
        # Use bf16 for Phi-3 to work with FlashAttention
        if "phi" in config["model_name"].lower():
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                offload_folder="./offload",
                attn_implementation="eager"  # Disable block sparse attention
            )
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=device_map
            )
        print("✓ Model and tokenizer loaded successfully")
        
        if config["use_wandb"] and wandb.run:
            wandb.log({
                "model/name": config["model_name"],
                "model/vocab_size": tokenizer.vocab_size,
                "config/device": str(device)
            })
            
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        sys.exit(1)
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = AbstractionTokenDataset(
            config["data_path"], 
            tokenizer, 
            max_length=config["max_length"],
            max_samples=config["max_samples"],
            model_name=config["model_name"]
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Skip token embedding resize for Phi-3 to avoid special token issues
        original_vocab_size = model.config.vocab_size
        if len(tokenizer) != original_vocab_size:
            print(f"⚠ Skipping token embedding resize (Phi-3 compatibility): {original_vocab_size} -> {len(tokenizer)}")
        else:
            print(f"✓ Token embeddings correct size: {len(tokenizer)}")
        
        # Analyze dataset composition
        abstain_samples = sum(1 for i in range(len(dataset)) if dataset[i]['should_abstain'])
        dont_abstain_samples = len(dataset) - abstain_samples
        
        print(f"  Abstain samples: {abstain_samples}")
        print(f"  Don't abstain samples: {dont_abstain_samples}")
        print(f"  Balance ratio: {dont_abstain_samples/len(dataset):.3f}")
        print()
        
        if config["use_wandb"] and wandb.run:
            wandb.log({
                "dataset/total_samples": len(dataset),
                "dataset/abstain_samples": abstain_samples,
                "dataset/dont_abstain_samples": dont_abstain_samples,
                "dataset/balance_ratio": dont_abstain_samples/len(dataset)
            })
            
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        sys.exit(1)
    
    # Setup LoRA
    print("Setting up LoRA configuration...")
    try:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("✓ LoRA configuration applied successfully")
        
    except Exception as e:
        print(f"✗ Error setting up LoRA: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        sys.exit(1)
    
    # Training
    print("Starting training...")
    try:
        trained_model = train_lora_abstention(model, dataset, tokenizer, config)
        
        # Save final model
        final_model_path = os.path.join(config["output_dir"], "final_model")
        trained_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print("✓ Training completed successfully")
        print(f"✓ Model saved to {final_model_path}")
        print()
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        sys.exit(1)
    
    # Interactive inference examples
    print("Interactive Inference Examples:")
    print("-" * 50)
    
    test_questions = [
        "What is 5 + 3?",
        "If I have 10 cookies and eat 4, how many are left?",
        "What is the derivative of x^2 + 3x + 1?",
        "Solve for x: 2x + 5 = 13"
    ]
    
    trained_model.eval()
    for i, question in enumerate(test_questions):
        print(f"Example {i+1}: {question}")
        
        try:
            # Format input
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    messages = [{"role": "user", "content": question}]
                    formatted_input = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    formatted_input = f"Q: {question}\nA:"
            else:
                formatted_input = f"Q: {question}\nA:"
            
            input_ids = tokenizer(formatted_input, return_tensors='pt')['input_ids'].to(trained_model.device)
            
            # Generate
            with torch.no_grad():
                outputs = trained_model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = generated_text[len(formatted_input):].strip()
            
            abstained = generated_part.startswith("<abstain>")
            dont_abstain = generated_part.startswith("<don't_abstain>")
            
            print(f"  Generated: {generated_part[:100]}...")
            print(f"  Abstained: {abstained}")
            print(f"  Don't Abstain: {dont_abstain}")
            
            if config["use_wandb"] and wandb.run:
                wandb.log({
                    f"inference/example_{i}/abstained": abstained,
                    f"inference/example_{i}/dont_abstain": dont_abstain
                })
                
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    print("=" * 60)
    print("Training and evaluation completed!")
    if config["use_wandb"] and wandb.run:
        if config["wandb_mode"] == "online":
            print(f"View full results at: {wandb.run.url}")
        wandb.finish()
    print("=" * 60)


if __name__ == "__main__":
    main()