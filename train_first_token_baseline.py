#!/usr/bin/env python3
"""
Training Script for First Token Baseline

This script demonstrates training a value head model that makes early stopping
decisions based only on the first output token, creating a baseline for comparison
with the full tokenwise approach.
"""

import torch
import sys
import wandb
import argparse
from transformers import AutoTokenizer, AutoConfig
from value_head_model import ValueHeadModel, TokenwiseValueHead, train_value_head
from datasets import MathReasoningDataset
from early_abstention import evaluate_abstention

override_config = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    "data_path": "gsm8k_mistral7b/samples_math_cot_multiple_2025-09-13T00-15-28.973445_train.jsonl",
    "output_dir": "gsm8k_mistral7b/output",
    "wandb_mode": "offline", 
    "device": "cuda"
}


def main():
    """Main training pipeline for first-token baseline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train first-token baseline value head")
    parser.add_argument("--model_name", type=str, default=None, help="Model name or path")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data JSONL")
    parser.add_argument("--output_dir", type=str, default=None, help="Base output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cuda:0, cpu, etc.)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("First Token Baseline Training")
    print("=" * 60)
    
    # Configuration for first-token baseline
    config = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        # "model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
        # "data_path": "samples_math_cot_multiple_2025-07-28T21-41-28.155556_train.jsonl",
        # "data_path": "one_sample_train.jsonl",
        "data_path": "rtp_dataset_train.jsonl",
        "output_dir": ".",
        "num_epochs": 2,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "max_length": 2048,
        "abstention_thresholds": [0.3, 0.5, 0.7],
        "use_wandb": True,
        "project_name": "first-token-baseline-rtp",
        "experiment_name": "first_token_only_rtp",
        "wandb_mode": "online",  # Change to "offline" for local logging only
        "save_model": True,  # Whether to save model artifacts to wandb
        "log_model_graph": True,  # Whether to log model architecture
        "log_frequency": 5,  # Log detailed examples every N batches
        "max_text_length": 1500,  # Maximum length of text to log to wandb
        "save_every_epoch": True,  # Save checkpoint after every epoch
        "resume_from_epoch": None,  # Set to epoch number to resume training
        "value_head_type": "tokenwise",  # Options: "tokenwise", "qwen3", "pooled_qwen3", "multilayer_qwen3"
        "attention_heads": 8,  # Number of attention heads for Qwen3-based value heads
        "num_key_value_heads": None,  # Number of key-value heads (None = same as attention_heads)
        "intermediate_size": None,  # MLP intermediate size (None = 4 * hidden_size)
        "num_layers": 2,  # Number of layers for multilayer value head
        "value_head_dropout": 0.1,  # Dropout rate for value head
        "hidden_act": "silu",  # Activation function for Qwen3 MLP
        "first_token_only": True,  # KEY: Enable first-token-only mode
        "device": "cuda:7"  # Device: "auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.
    }
    config = {**config, **override_config}
    
    # Override with CLI arguments (only if provided)
    config.update({k: v for k, v in vars(args).items() if k in config and v is not None})
    
    # Derive paths from output_dir
    output_dir = config.get("output_dir", ".")
    config["checkpoint_dir"] = f"{output_dir}/checkpoints_first_token"
    config["value_head_save_path"] = f"{output_dir}/trained_value_head_first_token.pth"
    
    print(f"Training Mode: {'First Token Only' if config['first_token_only'] else 'All Tokens'}")
    print(f"Model: {config['model_name']}")
    print(f"Data: {config['data_path']}")
    print(f"Device: {config['device']}")
    print()
    
    # Initialize wandb
    if config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            name=config["experiment_name"],
            config=config,
            mode=config["wandb_mode"],
            tags=["first-token", "baseline", "early-abstention"],
            notes="Baseline training using only first output token for early stopping decisions"
        )
        print(f"✓ W&B initialized ({config['wandb_mode']} mode)")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_config = AutoConfig.from_pretrained(config["model_name"], trust_remote_code=True)
        hidden_dim = model_config.hidden_size
        
        # Create value head and model (same architecture, different training)
        value_head = TokenwiseValueHead(hidden_dim)
        model = ValueHeadModel(
            model_name_or_path=config["model_name"],
            value_head=value_head,
            freeze_base_model=True,
            device=config["device"]
        )
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        sys.exit(1)
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = MathReasoningDataset(config["data_path"], tokenizer, max_length=config["max_length"])
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Show dataset composition
        correct_samples = sum(1 for i in range(len(dataset)) if dataset[i]['correctness'].item() > 0.5)
        print(f"  Correct samples: {correct_samples}")
        print(f"  Incorrect samples: {len(dataset) - correct_samples}")
        print()
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        return
    
    # Training with first-token-only mode
    print("Starting first-token baseline training...")
    try:
        train_value_head(
            model=model,
            dataset=dataset,
            num_epochs=config["num_epochs"],
            lr=config["learning_rate"],
            batch_size=config["batch_size"],
            use_wandb=config["use_wandb"],
            project_name=config["project_name"],
            log_frequency=config["log_frequency"],
            save_path=config["value_head_save_path"],
            save_every_epoch=config["save_every_epoch"],
            checkpoint_dir=config["checkpoint_dir"],
            resume_from_epoch=config["resume_from_epoch"],
            first_token_only=config["first_token_only"]  # KEY: Pass the baseline flag
        )
        print("✓ Training completed successfully")
        print()
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        return
    
    # Evaluation
    print("Evaluating first-token baseline performance...")
    try:
        results = evaluate_abstention(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            thresholds=config["abstention_thresholds"],
            use_wandb=config["use_wandb"]
        )
        
        print("First-Token Baseline Results:")
        print("-" * 50)
        for threshold, metrics in results.items():
            print(f"Threshold {threshold}:")
            print(f"  Coverage: {metrics['coverage']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Abstention Rate: {metrics['abstention_rate']:.3f}")
            print(f"  Avg Value: {metrics['avg_value']:.3f}")
            print()
            
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        return
    
    # Test inference examples
    print("Testing first-token baseline inference:")
    print("-" * 50)
    
    test_questions = [
        "What is 2 + 3?",
        "If I have 8 apples and eat 3, how many are left?"
    ]
    
    for question in test_questions:
        print(f"Q: {question}")
        try:
            # Format input
            formatted_input = f"Q: {question}\nA:"
            input_ids = tokenizer(formatted_input, return_tensors='pt')['input_ids']
            
            # Test with threshold 0.5
            generated, final_value = model.generate_with_abstention(
                input_ids, threshold=0.5, max_length=50, tokenizer=tokenizer
            )
            
            result = tokenizer.decode(generated[0], skip_special_tokens=True)
            abstained = final_value is not None and final_value < 0.5
            
            print(f"  Result: {'ABSTAINED' if abstained else 'GENERATED'}")
            print(f"  Value: {final_value:.3f}")
            if not abstained:
                generated_part = result[len(formatted_input):].strip()
                print(f"  Generated: {generated_part[:100]}...")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    print("=" * 60)
    print("First-token baseline training completed!")
    print(f"Final model saved to: {config['value_head_save_path']}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}/")
    if config["use_wandb"] and wandb.run:
        print(f"W&B logs: {wandb.run.url}")
        wandb.finish()
    print("=" * 60)


if __name__ == "__main__":
    main()