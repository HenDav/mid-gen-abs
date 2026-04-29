#!/usr/bin/env python3
"""
Training Example for Value Head Model

This script demonstrates how to train a transformer model with value head
for early abstention on mathematical reasoning tasks.
"""

import argparse
import sys
import torch
import wandb
from transformers import AutoTokenizer
from value_head_model import ValueHeadModel, TokenwiseValueHead, train_value_head
from datasets import MathReasoningDataset
from early_abstention import evaluate_abstention
from transformers import AutoConfig

mistral_config = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    "data_path": "gsm8k_mistral7b/samples_math_cot_multiple_2025-09-13T00-15-28.973445_train.jsonl",
    "wandb_mode": "offline", 
    "device": "cuda",
    "output_dir": "gsm8k_mistral7b/output"
}

# Override config - takes precedence over base config
override_config = mistral_config

def main():
    """Main training and evaluation pipeline with wandb tracking"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train tokenwise value head model")
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--data_path", type=str, help="Path to training data JSONL")
    parser.add_argument("--output_dir", type=str, help="Output directory (checkpoints and final model saved here)")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--max_length", type=int, help="Maximum sequence length")
    parser.add_argument("--device", type=str, help="Device to use")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Value Head Model Training Example with W&B Tracking")
    print("=" * 60)
    
    # Configuration
    config = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        # "model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
        # "data_path": "samples_math_cot_multiple_2025-07-28T21-41-28.155556_train.jsonl",
        # "data_path": "one_sample_train.jsonl",
        "data_path": "rtp_dataset_train.jsonl",
        "num_epochs": 5,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "max_length": 2048,
        "abstention_thresholds": [0.3, 0.5, 0.7],
        "use_wandb": True,
        "project_name": "value-head-abstention",
        "experiment_name": "tokenwise_value",
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
        "device": "cuda:6"  # Device: "auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.
    }
    config = {**config, **override_config}
    
    # Override config with command-line arguments (only those explicitly set)
    config.update({k: v for k, v in vars(args).items() if k in config and v is not None})
    
    # Derive paths from output_dir
    config["checkpoint_dir"] = f"{config['output_dir']}/checkpoints_our_method"
    config["value_head_save_path"] = f"{config['output_dir']}/trained_value_head.pth"
    
    # Initialize wandb
    if config["use_wandb"]:
        # Add device info to config for wandb tracking
        wandb_config = config.copy()
        
        wandb.init(
            project=config["project_name"],
            name=config["experiment_name"],
            config=wandb_config,
            mode=config["wandb_mode"],
            tags=["value-head", "abstention", "mathematical-reasoning", "qwen"],
            notes="Training transformer with value head for early abstention on math problems"
        )
        print(f"✓ W&B initialized ({config['wandb_mode']} mode)")
        if config["wandb_mode"] == "online":
            print(f"  View at: {wandb.run.url}")
        else:
            print(f"  Logging locally to: {wandb.run.dir}")
    
    print(f"Model: {config['model_name']}")
    print(f"Data: {config['data_path']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Max Length: {config['max_length']}")
    print()
    
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    
    # Configure device based on config
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
            if device.index is not None:
                torch.cuda.set_device(device.index)
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get model config to determine hidden dimension
        model_config = AutoConfig.from_pretrained(config["model_name"], trust_remote_code=True)
        hidden_dim = model_config.hidden_size
        
        # Create value head and model
        value_head = TokenwiseValueHead(hidden_dim)
        model = ValueHeadModel(
            model_name_or_path=config["model_name"],
            value_head=value_head,
            freeze_base_model=True,
            device=device
        )
            
        print("✓ Model and tokenizer loaded successfully")
        
        # Log model info to wandb (W&B automatically logs system stats)
        if config["use_wandb"] and wandb.run:
            wandb.log({
                "model/name": config["model_name"],
                "model/vocab_size": tokenizer.vocab_size,
                "model/pad_token": tokenizer.pad_token,
                "model/eos_token": tokenizer.eos_token,
                "config/device_requested": config["device"],
                "config/device_actual": str(device)
            })
            
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
        
        # Analyze dataset composition
        correct_samples = sum(1 for i in range(len(dataset)) if dataset[i]['correctness'].item() > 0.5)
        incorrect_samples = len(dataset) - correct_samples
        
        print(f"  Correct samples: {correct_samples}")
        print(f"  Incorrect samples: {incorrect_samples}")
        print(f"  Balance ratio: {correct_samples/len(dataset):.3f}")
        
        # Show sample data
        sample = dataset[0]
        sample_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"Sample text: {sample_text[:100]}...")
        print(f"Sample correctness: {sample['correctness'].item()}")
        print()
        
        # Log dataset info to wandb
        if config["use_wandb"] and wandb.run:
            wandb.log({
                "dataset/total_samples": len(dataset),
                "dataset/correct_samples": correct_samples,
                "dataset/incorrect_samples": incorrect_samples,
                "dataset/balance_ratio": correct_samples/len(dataset),
                "dataset/max_length": config["max_length"]
            })
            
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        sys.exit(1)
    
    # Training
    print("Starting training...")
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
            max_text_length=config["max_text_length"],
            save_path=config["value_head_save_path"],  # Automatically save value head weights
            save_every_epoch=config["save_every_epoch"],
            checkpoint_dir=config["checkpoint_dir"],
            resume_from_epoch=config["resume_from_epoch"]
        )
        print("✓ Training completed successfully")
        print()
    except Exception as e:
        print(f"✗ Error during training: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        sys.exit(1)
    
    # Demonstrate save/load functionality
    print("Demonstrating value head save/load functionality...")
    try:
        # Create a new model instance
        print("Creating new model instance...")
        new_value_head = TokenwiseValueHead(hidden_dim)
        new_model = ValueHeadModel(
            model_name_or_path=config["model_name"],
            value_head=new_value_head,
            freeze_base_model=True,
            device=device
        )
        
        # Load the saved value head weights
        print(f"Loading value head weights from {config['value_head_save_path']}...")
        new_model.load_value_head(config["value_head_save_path"])
        
        # Use the new model with loaded weights for evaluation
        model = new_model
        print("✓ Value head weights loaded successfully")
        print()
        
    except Exception as e:
        print(f"✗ Error during save/load demonstration: {e}")
        print("Continuing with original model...")
        print()
    
    # Evaluation
    print("Evaluating abstention performance...")
    try:
        results = evaluate_abstention(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            thresholds=config["abstention_thresholds"],
            use_wandb=config["use_wandb"]
        )
        
        print("Evaluation Results:")
        print("-" * 50)
        for threshold, metrics in results.items():
            print(f"Threshold {threshold}:")
            print(f"  Coverage: {metrics['coverage']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Abstention Rate: {metrics['abstention_rate']:.3f}")
            print(f"  Avg Value: {metrics['avg_value']:.3f}")
            print(f"  Value-Correctness Correlation: {metrics['value_correctness_correlation']:.3f}")
            print()
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        if config["use_wandb"] and wandb.run:
            wandb.finish()
        return
    
    # Interactive inference examples
    print("Interactive Inference Examples:")
    print("-" * 50)
    
    test_questions = [
        "What is 5 + 3?",
        "If I have 10 cookies and eat 4, how many are left?",
        "What is 2 * 6?"
    ]
    
    inference_results = []
    
    for question in test_questions:
        print(f"Input: Q: {question}\nA:")
        
        try:
            # Apply chat template if available
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    messages = [{"role": "user", "content": question}]
                    formatted_input = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    # Fallback to simple format
                    formatted_input = f"Q: {question}\nA:"
            else:
                formatted_input = f"Q: {question}\nA:"
            
            input_ids = tokenizer(formatted_input, return_tensors='pt')['input_ids']
            
            question_results = {"question": question}
            
            # Test with different thresholds
            for threshold in [0.3, 0.7]:
                generated, final_value = model.generate_with_abstention(
                    input_ids, threshold=threshold, max_length=100, tokenizer=tokenizer
                )
                
                result = tokenizer.decode(generated[0], skip_special_tokens=True)
                abstained = final_value is not None and final_value < threshold
                
                print(f"  Threshold {threshold}: {'ABSTAINED' if abstained else 'GENERATED'}")
                if not abstained:
                    # Show only the generated part
                    generated_part = result[len(question):].strip()
                    print(f"    Generated: {generated_part[:50]}...")
                print(f"    Final Value: {final_value:.3f}")
                
                question_results[f"threshold_{threshold}"] = {
                    "abstained": abstained,
                    "final_value": final_value,
                    "generated_text": result if not abstained else None
                }
                
        except Exception as e:
            print(f"    Error: {e}")
            question_results["error"] = str(e)
        
        inference_results.append(question_results)
        print()
    
    # Log inference examples to wandb
    if config["use_wandb"] and wandb.run:
        for i, result in enumerate(inference_results):
            if "error" not in result:
                for threshold_key, threshold_data in result.items():
                    if threshold_key.startswith("threshold_"):
                        threshold = threshold_key.split("_")[1]
                        wandb.log({
                            f"inference/example_{i}/question": result["question"],
                            f"inference/example_{i}/threshold_{threshold}/abstained": threshold_data["abstained"],
                            f"inference/example_{i}/threshold_{threshold}/final_value": threshold_data["final_value"]
                        })
    
    # Save model artifacts to wandb if requested
    if config["use_wandb"] and wandb.run and config.get("save_model", False):
        print("Saving model artifacts to W&B...")
        try:
            # Save model state dict
            model_artifact = wandb.Artifact(
                name=f"value-head-model-{wandb.run.id}",
                type="model",
                description="Trained value head model for early abstention"
            )
            
            # Save model weights (base model + value head)
            torch.save({
                'base_model': model.base_model.state_dict(),
                'value_head': model.value_head.state_dict()
            }, "model_weights.pth")
            model_artifact.add_file("model_weights.pth")
            
            # Save configuration
            import json
            with open("model_config.json", "w") as f:
                json.dump(config, f, indent=2)
            model_artifact.add_file("model_config.json")
            
            # Log the artifact
            wandb.log_artifact(model_artifact)
            print("✓ Model artifacts saved to W&B")
            
        except Exception as e:
            print(f"⚠️  Failed to save model artifacts: {e}")
    
    # Log final summary metrics
    if config["use_wandb"] and wandb.run:
        # Create summary table of all threshold results
        summary_data = []
        for threshold, metrics in results.items():
            summary_data.append([
                threshold,
                f"{metrics['coverage']:.3f}",
                f"{metrics['accuracy']:.3f}",
                f"{metrics['abstention_rate']:.3f}",
                f"{metrics['avg_value']:.3f}",
                f"{metrics['value_correctness_correlation']:.3f}"
            ])
        
        summary_table = wandb.Table(
            columns=["Threshold", "Coverage", "Accuracy", "Abstention Rate", "Avg Value", "Correlation"],
            data=summary_data
        )
        wandb.log({"evaluation_summary": summary_table})
        
        # Log final summary metrics
        wandb.summary.update({
            "best_threshold": max(results.keys(), key=lambda k: results[k]['accuracy']),
            "max_accuracy": max(metrics['accuracy'] for metrics in results.values()),
            "total_samples": len(dataset),
            "training_epochs": config["num_epochs"]
        })
    
    print("=" * 60)
    print("Training and evaluation completed!")
    if config["use_wandb"] and wandb.run:
        if config["wandb_mode"] == "online":
            print(f"View full results at: {wandb.run.url}")
        else:
            print(f"Local W&B logs saved to: {wandb.run.dir}")
        wandb.finish()
    print("=" * 60)


if __name__ == "__main__":
    # Run main pipeline
    main()