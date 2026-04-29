#!/usr/bin/env python3
"""
Create plots with abstention rate on x-axis instead of threshold
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import random
import os
import json
import sys
from datetime import datetime
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
from value_head_model import ValueHeadModel, TokenwiseValueHead
from datasets import MathReasoningDataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from peft import PeftModel

def calculate_prompt_based_token_savings(abstain_decisions, output_lengths):
    """
    Calculate token savings for prompt-based methods (LoRA, self-assessment, first-token baseline).
    When these methods abstain, they save ALL output tokens for that sample.
    
    Args:
        abstain_decisions: Boolean array indicating which samples were abstained
        output_lengths: Array of output token lengths for each sample
    
    Returns:
        float: Token savings rate (tokens saved / total output tokens)
    """
    abstain_decisions = np.array(abstain_decisions)
    output_lengths = np.array(output_lengths)
    
    total_output_tokens = np.sum(output_lengths)
    if total_output_tokens == 0:
        return 0.0
    
    # For prompt-based methods, abstention saves all tokens for that sample
    tokens_saved = np.sum(output_lengths[abstain_decisions])
    
    return tokens_saved / total_output_tokens

def calculate_tokenwise_token_savings(threshold, all_output_values, output_lengths):
    """
    Calculate token savings for tokenwise method.
    Tokenwise method stops generation at the first token below threshold.
    
    Args:
        threshold: Current threshold being evaluated
        all_output_values: List of output value arrays for each sample
        output_lengths: Array of output token lengths for each sample
    
    Returns:
        float: Token savings rate (tokens saved / total output tokens)
    """
    all_output_values = np.array(all_output_values, dtype=object)
    output_lengths = np.array(output_lengths)
    
    total_output_tokens = np.sum(output_lengths)
    if total_output_tokens == 0:
        return 0.0
    
    tokens_saved = 0
    
    for i, (output_values, output_length) in enumerate(zip(all_output_values, output_lengths)):
        if len(output_values) == 0:
            continue
            
        # Find first position where value < threshold (early stopping point)
        below_threshold = output_values < threshold
        stopping_positions = np.where(below_threshold)[0]
        
        if len(stopping_positions) > 0:
            # Stop at first position below threshold
            stopping_pos = stopping_positions[0]
            # Tokens saved = remaining tokens from stopping point to end
            tokens_saved += output_length - stopping_pos
    
    return tokens_saved / total_output_tokens

def plot_abstention_rate_analysis(model_name, data_path, baseline_path, full_model_path, lora_model_path,
                                 max_samples, assessment_prompt, device, batch_size, shuffle, seed, output_folder, x_range=None):
    """Create plots with abstention rate on x-axis"""
    
    print("=" * 60)
    print("ABSTENTION RATE ANALYSIS")
    print("=" * 60)
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"✓ Random seed set to: {seed}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    dataset = MathReasoningDataset(data_path, tokenizer, max_length=1024)
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    # Create subset dataset for analysis
    if shuffle:
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        subset_indices = all_indices[:n_samples]
        print(f"✓ Shuffled dataset indices for random sampling")
    else:
        subset_indices = list(range(n_samples))
    
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    # Create DataLoader
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    print(f"✓ Dataset loaded: {len(dataset)} samples, using {n_samples} for analysis")
    print(f"✓ DataLoader created with batch size: {batch_size}, shuffle: {shuffle}")
    
    def evaluate_model(model_path, model_name_str, first_token_only=False, self_assessment=False, lora_abstention=False, assessment_prompt="Will you correctly answer this question?"):
        """Evaluate a model across different thresholds"""
        print(f"\nEvaluating {model_name_str}...")
        
        if lora_abstention:
            # LoRA abstention model using abstention tokens
            print(f"Loading LoRA abstention model from: {model_path}")
            
            # Load the tokenizer that was saved with the LoRA model (already has abstention tokens)
            lora_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if lora_tokenizer.pad_token is None:
                lora_tokenizer.pad_token = lora_tokenizer.eos_token
            
            # Load base model
            if "phi" in model_name.lower():
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True, 
                    use_cache=False, 
                    attn_implementation="eager",
                    quantization_config=quantization_config
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_cache=False)
            
            # Resize model embeddings to match the saved tokenizer
            base_model.resize_token_embeddings(len(lora_tokenizer))
            print(f"Resized token embeddings to {len(lora_tokenizer)} tokens")
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Force eager attention for Phi models after PEFT loading
            if "phi" in model_name.lower():
                model.base_model.model.config.attn_implementation = "eager"
                
            model.to(device)
            model.eval()
            
            # Get abstention token IDs from the LoRA tokenizer using the same config as training
            from datasets import get_model_type, ABSTENTION_TOKENS
            model_type = get_model_type(model_name)
            token_config = ABSTENTION_TOKENS[model_type]
            
            abstain_token_id = lora_tokenizer.convert_tokens_to_ids(token_config["abstain"])
            dont_abstain_token_id = lora_tokenizer.convert_tokens_to_ids(token_config["dont_abstain"])
            
            print(f"Abstain token ID: {abstain_token_id}")
            print(f"Don't abstain token ID: {dont_abstain_token_id}")
            
            # Collect all values, labels, and output lengths using batched processing
            all_values = []
            all_abstention = []
            all_output_lengths = []
            
            for batch in tqdm(dataloader, desc=f"Processing {model_name_str}"):
                batch_question_texts = []
                batch_correctness = []
                
                # Prepare batch - extract ONLY questions (matching training)
                for item in batch:
                    input_ids = item['input_ids']
                    output_start_idx = item['output_start_idx']
                    
                    # Decode ONLY the question part (matching training data processing)
                    question_ids = input_ids[:output_start_idx]
                    question_text = tokenizer.decode(question_ids, skip_special_tokens=True)
                    
                    batch_question_texts.append(question_text)
                    batch_correctness.append(item['correctness'].item())
                
                # Tokenize questions only using the LoRA tokenizer (matching training input format)
                inputs = lora_tokenizer(batch_question_texts, return_tensors="pt",
                                      truncation=True, max_length=1024, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Get logits for next token prediction (at the end of question)
                    outputs = model(**inputs)
                    # Get logits for the LAST non-padded token in each sequence
                    batch_indices = torch.arange(len(batch_question_texts)).to(device)
                    last_token_positions = (inputs['attention_mask'].sum(dim=1) - 1).to(device)
                    next_token_logits = outputs.logits[batch_indices, last_token_positions, :]
                    
                    # Get logits for abstention tokens
                    abstain_logits = next_token_logits[:, abstain_token_id]
                    dont_abstain_logits = next_token_logits[:, dont_abstain_token_id]
                    
                    # Calculate abstention probabilities
                    logit_pairs = torch.stack([abstain_logits, dont_abstain_logits], dim=1)
                    abstain_probs = torch.softmax(logit_pairs, dim=1)[:, 0]  # Probability of abstain token
                    
                    
                all_values.extend(1 - abstain_probs.cpu().float().numpy())
                all_abstention.extend([1 - c for c in batch_correctness])  # Convert to should_abstain labels: 1=should_abstain (incorrect), 0=should_not_abstain (correct)
                
                # Calculate output lengths for token savings
                for item in batch:
                    input_ids = item['input_ids']
                    output_start_idx = item['output_start_idx']
                    output_length = len(input_ids) - output_start_idx
                    all_output_lengths.append(output_length)
            
            # Clean up model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        elif self_assessment:
            # Self-assessment baseline using yes/no logits
            print(f"Assessment prompt: '{assessment_prompt}'")
            
            # Load base model only
            if "phi" in model_name.lower():
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True, 
                    use_cache=False, 
                    attn_implementation="eager",
                    quantization_config=quantization_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_cache=False)
            model.to(device)
            model.eval()
            
            # Get yes/no token IDs
            yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = tokenizer.encode("no", add_special_tokens=False)[0]
            
            print(f"Yes token ID: {yes_token_id} ('{tokenizer.decode(yes_token_id)}')")
            print(f"No token ID: {no_token_id} ('{tokenizer.decode(no_token_id)}')")
            
            # Collect all values, labels, and output lengths using batched processing
            all_values = []
            all_abstention = []
            all_output_lengths = []
            
            for batch in tqdm(dataloader, desc=f"Processing {model_name_str}"):
                batch_question_texts = []
                batch_full_texts = []
                batch_correctness = []
                
                # Prepare batch - create assessment prompts
                for item in batch:
                    input_ids = item['input_ids']
                    output_start_idx = item['output_start_idx']
                    
                    # Decode ONLY the question part
                    question_ids = input_ids[:output_start_idx]
                    question_text = tokenizer.decode(question_ids, skip_special_tokens=True)
                    
                    # Create assessment prompt by appending the assessment question to the original question
                    assessment_text = question_text + " " + assessment_prompt
                    
                    batch_question_texts.append(question_text)
                    batch_full_texts.append(assessment_text)  # Use assessment prompt instead of full text
                    batch_correctness.append(item['correctness'].item())
                
                # Tokenize questions to find output start positions after padding
                question_inputs = tokenizer(batch_question_texts, return_tensors="pt",
                                          truncation=True, max_length=1024, padding=True)
                
                # Tokenize assessment prompts with padding
                inputs = tokenizer(batch_full_texts, return_tensors="pt", truncation=True,
                                 max_length=1024, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Calculate correct output start indices for padded sequences (end of assessment prompt)
                batch_start_index = []
                for i in range(len(batch_full_texts)):
                    # Find the actual length of assessment prompt tokens (excluding padding)
                    assessment_length = (inputs['attention_mask'][i] == 1).sum().item()
                    batch_start_index.append(assessment_length)
                
                with torch.no_grad():
                    # Get logits for next token prediction (at the end of assessment prompt)
                    outputs = model(**inputs)
                    next_token_logits = outputs.logits[torch.arange(len(batch)).to(device), torch.LongTensor(batch_start_index).to(device)-1, :]  # Logits for next token after assessment prompt
                    
                    # Get logits for yes/no tokens
                    yes_logits = next_token_logits[:, yes_token_id]
                    no_logits = next_token_logits[:, no_token_id]
                    
                    # Calculate confidence scores - use "no" probability as confidence (higher = more confident = less likely to abstain)
                    logit_pairs = torch.stack([yes_logits, no_logits], dim=1)
                    yes_probs = torch.softmax(logit_pairs, dim=1)[:, 1]  # Probability of "no" (confident)
                
                all_values.extend(yes_probs.cpu().float().numpy())
                all_abstention.extend([1 - c for c in batch_correctness])  # Convert to should_abstain labels: 1=should_abstain (incorrect), 0=should_not_abstain (correct)
                
                # Calculate output lengths for token savings
                for item in batch:
                    input_ids = item['input_ids']
                    output_start_idx = item['output_start_idx']
                    output_length = len(input_ids) - output_start_idx
                    all_output_lengths.append(output_length)
            
            # Clean up model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        else:
            # Value head models (baseline and full)
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            hidden_dim = config.hidden_size
            
            value_head = TokenwiseValueHead(hidden_dim)
            model = ValueHeadModel(
                model_name_or_path=model_name,
                value_head=value_head,
                freeze_base_model=True,
                device=device
            )
            model.load_value_head(model_path)
            model.eval()
            
            # Collect all values, labels, and output lengths (value head models process individually)
            all_values = []
            all_abstention = []
            all_output_lengths = []
            all_output_values = []  # For tokenwise method, store individual output value arrays
            
            for batch in tqdm(dataloader, desc=f"Processing {model_name_str}"):
                for item in batch:  # Process items individually for value head models
                    input_ids = item['input_ids'].unsqueeze(0).to(device)
                    attention_mask = torch.ones_like(input_ids)
                    output_start_idx = item['output_start_idx']
                    should_abstain = 1 - item['correctness'].item()  # Convert to should_abstain label: 1=should_abstain (incorrect), 0=should_not_abstain (correct)
                    
                    with torch.no_grad():
                        values = model(input_ids=input_ids, attention_mask=attention_mask)
                        value_probs = torch.sigmoid(values)
                    
                    # Extract output token values
                    # For individual processing, output_start_idx is already correct for the sequence
                    seq_length = input_ids.size(1)
                    if output_start_idx < seq_length:
                        output_indices = torch.arange(output_start_idx, seq_length)
                    else:
                        output_indices = torch.tensor([])
                    
                    if len(output_indices) > 0:
                        output_values = value_probs[0, output_indices].cpu().float().numpy()
                        
                        if first_token_only:
                            # Use only first token for baseline
                            decision_value = output_values[0]
                        else:
                            # Use minimum value for full model
                            decision_value = output_values.min()
                        
                        all_values.append(decision_value)
                        all_abstention.append(should_abstain)  # Use as abstention labels (1=should_abstain (incorrect), 0=should_not_abstain (correct))
                        
                        # Store output length and values for token savings calculation
                        all_output_lengths.append(len(output_values))
                        if not first_token_only:
                            # For tokenwise method, store the full output value array
                            all_output_values.append(output_values)
                        else:
                            # For first-token baseline, we don't need individual values
                            all_output_values.append(np.array([decision_value]))
            
            # Clean up model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        all_values = np.array(all_values)
        all_abstention = np.array(all_abstention)
        all_output_lengths = np.array(all_output_lengths)
        
        # Evaluate across different thresholds using actual test values
        # Use the actual values from the test data as thresholds, plus 0 and 1 as boundaries
        unique_values = np.unique(all_values)
        thresholds = np.concatenate([[0.0], unique_values, [1.0]])
        thresholds = np.unique(thresholds)  # Remove duplicates and sort
        
        results = {
            'thresholds': [],
            'abstention_rates': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'balanced_accuracies': [],
            'token_savings_rates': []
        }
        
        print(f"  Using {len(thresholds)} threshold points based on actual test values")
        
        for threshold in thresholds:
            # Abstention decisions
            # For all models: abstain when value/probability is LOW (below threshold)
            # all_abstention contains: 1=should_abstain (incorrect), 0=should_not_abstain (correct)
            abstain_decisions = all_values < threshold
            abstention_rate = np.mean(abstain_decisions)
            
            # Calculate metrics (handle edge cases properly)
            try:
                # Check if we have varied predictions to avoid sklearn warnings
                unique_true = len(np.unique(all_abstention))
                unique_pred = len(np.unique(abstain_decisions))
                
                if unique_true == 1 or unique_pred == 1:
                    # Handle case where all labels are the same
                    if unique_true == 1 and unique_pred == 1:
                        # Both true and predicted have only one class
                        if all_abstention[0] == abstain_decisions[0]:
                            precision = recall = f1 = balanced_acc = 1.0
                        else:
                            precision = recall = f1 = balanced_acc = 0.0
                    else:
                        # One has variety, the other doesn't
                        precision = recall = f1 = balanced_acc = 0.0
                else:
                    # Normal case with varied labels
                    precision = precision_score(all_abstention, abstain_decisions, zero_division=0, labels=[0, 1])
                    recall = recall_score(all_abstention, abstain_decisions, zero_division=0, labels=[0, 1])
                    f1 = f1_score(all_abstention, abstain_decisions, average="macro", zero_division=0, labels=[0, 1])
                    balanced_acc = balanced_accuracy_score(all_abstention, abstain_decisions)
                
                # Calculate token savings rate based on method type
                if lora_abstention or self_assessment or first_token_only:
                    # Prompt-based methods and first-token baseline: abstention saves all output tokens
                    token_savings_rate = calculate_prompt_based_token_savings(abstain_decisions, all_output_lengths)
                else:
                    # Tokenwise method: calculate position-dependent savings
                    token_savings_rate = calculate_tokenwise_token_savings(threshold, all_output_values, all_output_lengths)
                
                results['thresholds'].append(threshold)
                results['abstention_rates'].append(abstention_rate)
                results['precisions'].append(precision)
                results['recalls'].append(recall)
                results['f1_scores'].append(f1)
                results['balanced_accuracies'].append(balanced_acc)
                results['token_savings_rates'].append(token_savings_rate)
                
            except Exception as e:
                # Only skip on actual calculation errors
                print(f"Warning: Error calculating metrics for threshold {threshold}: {e}")
                continue
        
        # Return results along with trajectory values for CSV output
        if not (lora_abstention or self_assessment or first_token_only):
            # For full tokenwise model only, return the full output values for trajectory analysis
            return results, all_values, all_abstention, all_output_lengths, all_output_values
        else:
            # For other models, no token-wise trajectories needed
            return results, all_values, all_abstention, all_output_lengths, None
    
    def plot_metric(ax, results_dict, metric_key, metric_name, x_axis='abstention_rates', x_label='Abstention Rate', x_range=None):
        """Helper function to plot a single metric"""
        model_configs = [
            ('full', 'Mid-Generation Abs. using Hidden States', 'bo-', 4, 1.0),
            ('baseline', 'Prompt-Based Abs. using Hidden States', 'r^--', 6, 0.8),
            ('self_assessment', 'Self-Assessment', 'gs-', 4, 0.8),
            ('lora_abstention', 'LoRA Abstention', 'md-', 5, 0.9)
        ]
        
        for model_key, label, style, markersize, alpha in model_configs:
            results = results_dict[model_key]
            if len(results[x_axis]) > 0:
                ax.plot(results[x_axis], results[metric_key],
                       style, label=label, linewidth=2, markersize=markersize, alpha=alpha)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs {x_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis range
        if x_range is not None:
            ax.set_xlim(x_range[0], x_range[1])
        else:
            ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Evaluate all models and store trajectory values
    results = {}
    trajectory_data = {}
    
    model_configs = [
        ('lora_abstention', lora_model_path, "LoRA Abstention Model", {'lora_abstention': True}),
        ('self_assessment', None, "Self-Assessment Baseline", {'self_assessment': True, 'assessment_prompt': assessment_prompt}),
        ('baseline', baseline_path, "First-Token Baseline", {'first_token_only': True}),
        ('full', full_model_path, "Full Tokenwise Model", {'first_token_only': False}),
    ]
    
    for model_key, model_path, model_name_str, kwargs in model_configs:
        model_results, all_values, all_abstention, all_output_lengths, all_output_values = evaluate_model(model_path, model_name_str, **kwargs)
        results[model_key] = model_results
        trajectory_data[model_key] = {
            'values': all_values,
            'labels': all_abstention,
            'output_lengths': all_output_lengths,
            'output_values': all_output_values  # Full trajectories only for tokenwise method
        }
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"✓ Output folder created: {output_folder}")
    
    # Save run arguments
    run_info = {
        'timestamp': datetime.now().isoformat(),
        'command_line': ' '.join(sys.argv),
        'arguments': {
            'model_name': model_name,
            'data_path': data_path,
            'baseline_path': baseline_path,
            'full_model_path': full_model_path,
            'lora_model_path': lora_model_path,
            'max_samples': max_samples,
            'assessment_prompt': assessment_prompt,
            'device': device,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'seed': seed,
            'output_folder': output_folder,
            'x_range': x_range
        }
    }
    
    args_filename = os.path.join(output_folder, "run_args.json")
    with open(args_filename, 'w') as f:
        json.dump(run_info, f, indent=2)
    print(f"✓ Run arguments saved to: {args_filename}")
    
    # Save trajectory values to CSV (always)
    csv_data = []
    
    for model_name, traj_data in trajectory_data.items():
        values = traj_data['values']
        labels = traj_data['labels']
        output_lengths = traj_data['output_lengths']
        output_values = traj_data['output_values']
        
        if model_name == 'full' and output_values is not None:
            # For tokenwise method only: save full token-by-token trajectories
            for sample_idx, (decision_value, label, output_length, trajectory) in enumerate(zip(values, labels, output_lengths, output_values)):
                for token_idx, token_value in enumerate(trajectory):
                    csv_data.append({
                        'model': model_name,
                        'sample_index': sample_idx,
                        'token_index': token_idx,
                        'trajectory_value': token_value,
                        'decision_value': decision_value,
                        'should_abstain_label': label,
                        'output_length': output_length
                    })
        else:
            # For all other models: only save the aggregated decision value
            for sample_idx, (value, label, output_length) in enumerate(zip(values, labels, output_lengths)):
                csv_data.append({
                    'model': model_name,
                    'sample_index': sample_idx,
                    'token_index': 0,  # Single decision value
                    'trajectory_value': value,
                    'decision_value': value,
                    'should_abstain_label': label,
                    'output_length': output_length
                })
    
    df = pd.DataFrame(csv_data)
    csv_filename = os.path.join(output_folder, "trajectory_values.csv")
    df.to_csv(csv_filename, index=False)
    print(f"✓ Trajectory values saved to CSV: {csv_filename}")
    print(f"  - Total rows: {len(csv_data)}")
    print(f"  - Full tokenwise trajectories saved for 'full' model only")
    
    # Create plots with abstention rate on x-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot all metrics using helper function
    plot_metric(ax1, results, 'precisions', 'Precision', x_range=x_range)
    plot_metric(ax2, results, 'recalls', 'Recall', x_range=x_range)
    plot_metric(ax3, results, 'f1_scores', 'Balanced F1 Score', x_range=x_range)
    plot_metric(ax4, results, 'balanced_accuracies', 'Balanced Accuracy', x_range=x_range)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(output_folder, 'abstention_rate_analysis.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_filename}")
    
    # Create additional plots with token savings rate on x-axis
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot all metrics with token savings rate on x-axis
    plot_metric(ax5, results, 'precisions', 'Precision', 'token_savings_rates', 'Token Savings Rate', x_range=x_range)
    plot_metric(ax6, results, 'recalls', 'Recall', 'token_savings_rates', 'Token Savings Rate', x_range=x_range)
    plot_metric(ax7, results, 'f1_scores', 'Balanced F1 Score', 'token_savings_rates', 'Token Savings Rate', x_range=x_range)
    plot_metric(ax8, results, 'balanced_accuracies', 'Balanced Accuracy', 'token_savings_rates', 'Token Savings Rate', x_range=x_range)
    
    plt.tight_layout()
    
    # Save token savings rate plot
    token_plot_filename = os.path.join(output_folder, 'token_savings_rate_analysis.png')
    plt.savefig(token_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Token savings rate plot saved to: {token_plot_filename}")
    
    def print_model_stats(name, results):
        """Helper function to print model statistics"""
        if len(results['abstention_rates']) > 0:
            print(f"\n{name}:")
            print(f"  Abstention rate range: {min(results['abstention_rates']):.3f} - {max(results['abstention_rates']):.3f}")
            print(f"  Max balanced F1 score: {max(results['f1_scores']):.3f}")
            print(f"  Max balanced accuracy: {max(results['balanced_accuracies']):.3f}")
        else:
            print(f"\n{name}:")
            print(f"  No meaningful abstention range (values outside tested thresholds)")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print_model_stats("Full Tokenwise Model", results['full'])
    print_model_stats("First-Token Baseline", results['baseline'])
    print_model_stats("Self-Assessment Baseline", results['self_assessment'])
    print_model_stats("LoRA Abstention Model", results['lora_abstention'])
    
    plt.show()
    
    # Show the token savings rate plot as well
    plt.figure(fig2.number)
    plt.show()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot abstention rate analysis comparing multiple abstention models')
    
    # Model and data configuration
    parser.add_argument('--model-name', type=str,
                       default="Qwen/Qwen2.5-Math-7B-Instruct",
                       help='Hugging Face model name or path')
    parser.add_argument('--data-path', type=str,
                       default="samples_math_cot_multiple_2025-07-28T21-41-28.155556_test.jsonl",
                       help='Path to the test dataset JSONL file')
    parser.add_argument('--baseline-path', type=str,
                       default="checkpoints_first_token/value_head_epoch_3.pth",
                       help='Path to the first-token baseline checkpoint')
    parser.add_argument('--full-model-path', type=str,
                       default="checkpoints_tokenwise_value/value_head_epoch_5.pth",
                       help='Path to the full tokenwise model checkpoint')
    parser.add_argument('--lora-model-path', type=str,
                       default='./lora_best_hparams/final_model',
                       help='Path to the LoRA abstention model checkpoint (optional)')
    
    # Evaluation configuration
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all samples)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for processing (default: 4)')
    parser.add_argument('--shuffle', action='store_true',
                       help='Randomly shuffle dataset indices for subset selection')
    parser.add_argument('--assessment-prompt', type=str,
                       default="Will you incorrectly answer this question? (answer yes/no)",
                       help='Prompt to append for self-assessment baseline')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). If not specified, auto-detects')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (optional)')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Output folder name for plots and CSV file')
    parser.add_argument('--x-range', type=float, nargs=2, default=[0.0, 0.4],
                       help='X-axis range for plots [min, max] (default: [0.0, 0.4])')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    plot_abstention_rate_analysis(
        model_name=args.model_name,
        data_path=args.data_path,
        baseline_path=args.baseline_path,
        full_model_path=args.full_model_path,
        lora_model_path=args.lora_model_path,
        max_samples=args.max_samples,
        assessment_prompt=args.assessment_prompt,
        device=device,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        seed=args.seed,
        output_folder=args.output_folder,
        x_range=args.x_range
    )