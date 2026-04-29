

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict
from tqdm import tqdm
import logging

# Abstention token configuration
ABSTENTION_TOKENS = {
    "phi": {
        "abstain": "<|dummy_id_0|>",
        "dont_abstain": "<|dummy_id_1|>", 
        "add_to_tokenizer": False  # These tokens already exist in Phi-3
    },
    "default": {
        "abstain": "<abstain>",
        "dont_abstain": "<don't_abstain>",
        "add_to_tokenizer": True  # Need to add these tokens
    }
}

def get_model_type(model_name_or_path):
    """Determine model type from name/path"""
    if "phi" in model_name_or_path.lower():
        return "phi"
    return "default"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathReasoningDataset(Dataset):
    """JSONL dataset for mathematical reasoning with correctness labels"""
    
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            jsonl_path (str): Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
        """
        with open(jsonl_path) as f:
            raw_data = [json.loads(line) for line in f]
        
        # Expand dataset to include all response-correctness pairs
        self.data = []
        for item in raw_data:
            question = item['doc']['question']
            
            # Handle different response structures
            if item['resps'] and len(item['resps']) > 0:
                responses = item['resps'][0] if isinstance(item['resps'][0], list) else item['resps']
                correctness_values = item['doc']['correctness']
                
                # Create one training example for each response-correctness pair
                for resp_idx, (response, correctness) in enumerate(zip(responses, correctness_values)):
                    self.data.append({
                        'question': question,
                        'response': response,
                        'correctness': correctness,
                        'original_idx': len(self.data),  # Track original item
                        'response_idx': resp_idx  # Track which response this is
                    })
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Expanded dataset: {len(raw_data)} original items -> {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single data item with output token boundaries.
        
        Args:
            idx (int): Item index
            
        Returns:
            dict: Tokenized input with correctness label and output token start position
        """
        item = self.data[idx]
        
        question = item['question']
        answer = item['response']
        
        # Tokenize question and answer separately to find output token boundaries
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Format as chat messages
            messages_question = [{"role": "user", "content": question}]
            messages_full = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            try:
                question_text = self.tokenizer.apply_chat_template(
                    messages_question,
                    tokenize=False,
                    add_generation_prompt=True  # This adds the assistant prompt
                )
                full_text = self.tokenizer.apply_chat_template(
                    messages_full,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except:
                # Fallback to simple format if chat template fails
                question_text = f"Q: {question}\nA:"
                full_text = f"Q: {question}\nA: {answer}"
        else:
            # Simple format for models without chat template
            question_text = f"Q: {question}\nA:"
            full_text = f"Q: {question}\nA: {answer}"
        
        # Tokenize question part to find where output tokens start
        question_tokens = self.tokenizer(
            question_text,
            max_length=self.max_length,
            padding=False,
            truncation=False,
            return_tensors='pt'
        )
        
        # Tokenize full sequence
        full_tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=False,
            truncation=False,
            return_tensors='pt'
        )
        
        # Find where output tokens start (after question + prompt)
        output_start_idx = question_tokens['input_ids'].size(1)
        
        return {
            'input_ids': full_tokens['input_ids'].squeeze(),
            'attention_mask': full_tokens['attention_mask'].squeeze(),
            'correctness': torch.tensor(float(item['correctness'])),
            'output_start_idx': output_start_idx  # Where output tokens begin
        }


def calculate_dataset_threshold(dataset):
    """
    Calculate the mean correctness from dataset samples to use as threshold.
    
    Args:
        dataset: MathReasoningDataset instance
        
    Returns:
        float: Mean correctness value to use as threshold
    """
    correctness_values = []
    for i in range(len(dataset)):
        correctness_values.append(dataset[i]['correctness'].item())
    
    mean_correctness = np.mean(correctness_values)
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(correctness_values)}")
    print(f"  Positive samples: {sum(correctness_values)}")
    print(f"  Mean correctness: {mean_correctness:.4f}")
    print(f"  Using {mean_correctness:.4f} as threshold for binary classification")
    
    return mean_correctness

class AbstractionTokenDataset(Dataset):
    """Dataset for training abstention token prediction with separate input/output"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, max_samples: int = None, model_name: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        
        # Load original dataset
        self.original_dataset = MathReasoningDataset(data_path, tokenizer, max_length)
        
        # Get model-specific abstention tokens
        model_identifier = model_name or tokenizer.name_or_path or ""
        model_type = get_model_type(model_identifier)
        token_config = ABSTENTION_TOKENS[model_type]
        
        self.abstain_token = token_config["abstain"]
        self.dont_abstain_token = token_config["dont_abstain"]
        
        if token_config["add_to_tokenizer"]:
            # Add tokens for models that need it (Qwen, etc.)
            special_tokens = {
                "additional_special_tokens": [self.abstain_token, self.dont_abstain_token]
            }
            num_added = tokenizer.add_special_tokens(special_tokens)
            logger.info(f"Added {num_added} new tokens to tokenizer")
        
        # Always use convert_tokens_to_ids for consistency
        self.abstain_token_id = tokenizer.convert_tokens_to_ids(self.abstain_token)
        self.dont_abstain_token_id = tokenizer.convert_tokens_to_ids(self.dont_abstain_token)
        
        logger.info(f"Using {model_type} tokens: {self.abstain_token}={self.abstain_token_id}, {self.dont_abstain_token}={self.dont_abstain_token_id}")
        
        # Prepare training examples
        self.examples = self._prepare_training_examples()
        
    def _prepare_training_examples(self) -> List[Dict]:
        """Prepare training examples with proper sequence alignment for abstention token prediction"""
        examples = []
        
        # Determine how many samples to process
        total_samples = len(self.original_dataset)
        n_samples = min(self.max_samples, total_samples) if self.max_samples else total_samples
        
        for i in tqdm(range(n_samples), desc="Preparing training examples"):
            item = self.original_dataset[i]
            
            # Get original data
            input_ids = item['input_ids']
            output_start_idx = item['output_start_idx']
            correctness = item['correctness'].item()
            
            # Split into question and answer parts
            question_ids = input_ids[:output_start_idx]
            answer_ids = input_ids[output_start_idx:]
            
            # Determine abstention decision based on correctness
            should_abstain = correctness == 0  # Abstain if incorrect
            abstention_token_id = self.abstain_token_id if should_abstain else self.dont_abstain_token_id
            
            # FIXED: Create proper training sequence
            # Input: question only (what the model sees)
            input_sequence = question_ids
            
            # Target: abstention_token + original_answer (what we want to predict)
            # But we only train on the abstention token prediction
            target_output = torch.tensor([abstention_token_id], dtype=torch.long)
            
            # Truncate input if necessary
            max_input_length = self.max_length - 10  # Reserve space for target
            if len(input_sequence) > max_input_length:
                input_sequence = input_sequence[:max_input_length]
            
            # Create attention masks
            input_attention_mask = torch.ones_like(input_sequence)
            
            # FIXED: Train only on the abstention token
            output_attention_mask = torch.ones_like(target_output)  # Train on abstention token
            
            examples.append({
                'input_ids': input_sequence,
                'input_attention_mask': input_attention_mask,
                'target_output_ids': target_output,
                'target_attention_mask': output_attention_mask,
                'abstention_token_id': abstention_token_id,
                'should_abstain': should_abstain,
                'original_correctness': correctness,
                'output_start_idx': len(input_sequence),  # Where output should start
            })
        
        logger.info(f"Prepared {len(examples)} training examples")
        
        # Print some statistics
        abstain_count = sum(1 for ex in examples if ex['should_abstain'])
        dont_abstain_count = len(examples) - abstain_count
        logger.info(f"Examples with abstain token: {abstain_count}")
        logger.info(f"Examples with don't abstain token: {dont_abstain_count}")
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class AbstractionTokenDataCollator:
    """Custom data collator for abstention token training with simplified target structure"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        # Separate inputs and targets
        input_ids = [item['input_ids'] for item in batch]
        input_attention_masks = [item['input_attention_mask'] for item in batch]
        target_output_ids = [item['target_output_ids'] for item in batch]
        target_attention_masks = [item['target_attention_mask'] for item in batch]
        
        # Pad inputs
        max_input_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_input_masks = []
        
        for ids, mask in zip(input_ids, input_attention_masks):
            padding_length = max_input_len - len(ids)
            padded_ids = torch.cat([ids, torch.full((padding_length,), self.tokenizer.pad_token_id)])
            padded_mask = torch.cat([mask, torch.zeros(padding_length)])
            padded_input_ids.append(padded_ids)
            padded_input_masks.append(padded_mask)
        
        # FIXED: Handle simplified target structure (single abstention token)
        max_target_len = max(len(ids) for ids in target_output_ids) if target_output_ids else 1
        max_target_len = max(max_target_len, 1)  # Ensure at least 1 token
        padded_target_ids = []
        padded_target_masks = []
        
        for ids, mask in zip(target_output_ids, target_attention_masks):
            padding_length = max_target_len - len(ids)
            if padding_length > 0:
                padded_ids = torch.cat([ids, torch.full((padding_length,), self.tokenizer.pad_token_id)])
                padded_mask = torch.cat([mask, torch.zeros(padding_length)])
            else:
                padded_ids = ids
                padded_mask = mask
            padded_target_ids.append(padded_ids)
            padded_target_masks.append(padded_mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_input_masks),
            'target_ids': torch.stack(padded_target_ids),
            'target_attention_mask': torch.stack(padded_target_masks),
            'abstention_token_ids': torch.tensor([item['abstention_token_id'] for item in batch]),
            'should_abstain': torch.tensor([item['should_abstain'] for item in batch]),
        }
