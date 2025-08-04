"""
Data loading and preprocessing for the Alchemist Architecture.

Handles the curated datasets for each cognitive domain with proper
formatting and sharding for training specialists.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from alchemist.foundation.tokenizer import SharedTokenizer


@dataclass
class TrainingExample:
    """A single training example with prompt and completion."""
    prompt: str
    completion: str
    domain: str  # 'math', 'creative', 'general'
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "domain": self.domain
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'TrainingExample':
        """Create from dictionary format."""
        return cls(
            prompt=data["prompt"],
            completion=data["completion"],
            domain=data["domain"]
        )


class AlchemistDataset(Dataset):
    """
    Dataset for Alchemist Architecture training.
    
    Handles multiple domains with unified tokenization and
    proper formatting for specialist training.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: SharedTokenizer,
        max_length: int = 512,
        domain_filter: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSONL data file
            tokenizer: Shared tokenizer instance
            max_length: Maximum sequence length
            domain_filter: Optional domain filter ('math', 'creative', 'general')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domain_filter = domain_filter
        
        # Load examples
        self.examples = self._load_examples(data_path)
        
        # Tokenize all examples
        self.tokenized_examples = self._tokenize_examples()
    
    def _load_examples(self, data_path: str) -> List[TrainingExample]:
        """Load examples from JSONL file."""
        examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    example = TrainingExample.from_dict(data)
                    
                    # Apply domain filter if specified
                    if self.domain_filter is None or example.domain == self.domain_filter:
                        examples.append(example)
        
        return examples
    
    def _tokenize_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Tokenize all examples and prepare for training."""
        tokenized = []
        
        for example in self.examples:
            # Combine prompt and completion
            full_text = f"{example.prompt}{example.completion}"
            
            # Tokenize
            tokens = self.tokenizer.encode(full_text)
            
            # Truncate if necessary
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            # Convert to tensor
            input_ids = torch.tensor(tokens, dtype=torch.long)
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            tokenized.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "domain": example.domain
            })
        
        return tokenized
    
    def __len__(self) -> int:
        return len(self.tokenized_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.tokenized_examples[idx]


class DataCurator:
    """
    Data curation utilities for preparing domain-specific datasets.
    
    Handles the creation of the curated mini-datasets specified in Phase 0.2.
    """
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data curator.
        
        Args:
            output_dir: Output directory for curated datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_gsm8k_math_dataset(self, source_path: str, target_size: int = 10000) -> str:
        """
        Prepare GSM8K math dataset.
        
        Args:
            source_path: Path to GSM8K dataset
            target_size: Target number of problems
            
        Returns:
            Path to the prepared dataset
        """
        import datasets
        
        # Load GSM8K
        dataset = datasets.load_dataset("gsm8k", "main")
        
        # Filter out "draw picture" prompts and deduplicate
        filtered_examples = []
        seen_problems = set()
        
        for example in dataset["train"]:
            question = example["question"]
            
            # Skip if contains drawing instructions
            if "draw" in question.lower() or "picture" in question.lower():
                continue
            
            # Skip if duplicate
            if question in seen_problems:
                continue
            
            seen_problems.add(question)
            
            # Format for training
            prompt = f"Question: {question}\nAnswer:"
            completion = f" {example['answer']}"
            
            example_dict = {
                "prompt": prompt,
                "completion": completion,
                "domain": "math"
            }
            
            filtered_examples.append(example_dict)
            
            if len(filtered_examples) >= target_size:
                break
        
        # Split into train/val
        random.shuffle(filtered_examples)
        split_idx = int(0.9 * len(filtered_examples))
        
        train_examples = filtered_examples[:split_idx]
        val_examples = filtered_examples[split_idx:]
        
        # Save datasets
        train_path = os.path.join(self.output_dir, "math", "train.jsonl")
        val_path = os.path.join(self.output_dir, "math", "val.jsonl")
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        
        self._save_jsonl(train_examples, train_path)
        self._save_jsonl(val_examples, val_path)
        
        print(f"Math dataset prepared:")
        print(f"  Train: {len(train_examples)} examples")
        print(f"  Val: {len(val_examples)} examples")
        print(f"  Total: {len(filtered_examples)} examples")
        
        return train_path
    
    def prepare_creative_dataset(self, source_paths: List[str], target_size: int = 6000) -> str:
        """
        Prepare creative writing dataset.
        
        Args:
            source_paths: Paths to source creative writing data
            target_size: Target number of examples
            
        Returns:
            Path to the prepared dataset
        """
        # This would load from public domain books, fan fiction, etc.
        # For now, create synthetic examples for demonstration
        
        creative_examples = []
        
        # Generate synthetic creative writing examples
        prompts = [
            "Write a short story about",
            "Create a poem about",
            "Describe a magical world where",
            "Tell me about a character who",
            "Write a dialogue between"
        ]
        
        themes = [
            "a robot learning to paint",
            "a library that comes alive at night",
            "a time traveler's first day in the past",
            "a chef who can taste emotions",
            "a musician who hears colors"
        ]
        
        for i in range(target_size):
            prompt_base = random.choice(prompts)
            theme = random.choice(themes)
            
            prompt = f"{prompt_base} {theme}."
            completion = f" Here is a creative response about {theme}..."
            
            example_dict = {
                "prompt": prompt,
                "completion": completion,
                "domain": "creative"
            }
            
            creative_examples.append(example_dict)
        
        # Split into train/val
        random.shuffle(creative_examples)
        split_idx = int(0.9 * len(creative_examples))
        
        train_examples = creative_examples[:split_idx]
        val_examples = creative_examples[split_idx:]
        
        # Save datasets
        train_path = os.path.join(self.output_dir, "creative", "train.jsonl")
        val_path = os.path.join(self.output_dir, "creative", "val.jsonl")
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        
        self._save_jsonl(train_examples, train_path)
        self._save_jsonl(val_examples, val_path)
        
        print(f"Creative dataset prepared:")
        print(f"  Train: {len(train_examples)} examples")
        print(f"  Val: {len(val_examples)} examples")
        print(f"  Total: {len(creative_examples)} examples")
        
        return train_path
    
    def prepare_general_dataset(self, source_paths: List[str], target_size: int = 5000) -> str:
        """
        Prepare general domain dataset.
        
        Args:
            source_paths: Paths to general domain data
            target_size: Target number of examples
            
        Returns:
            Path to the prepared dataset
        """
        # This would load from CC-Net, Pile, etc.
        # For now, create synthetic examples
        
        general_examples = []
        
        # Generate synthetic general examples
        topics = [
            "explain how photosynthesis works",
            "describe the history of the internet",
            "compare different programming paradigms",
            "discuss the benefits of exercise",
            "explain machine learning concepts"
        ]
        
        for i in range(target_size):
            topic = random.choice(topics)
            
            prompt = f"Please explain: {topic}"
            completion = f" Here is an explanation of {topic}..."
            
            example_dict = {
                "prompt": prompt,
                "completion": completion,
                "domain": "general"
            }
            
            general_examples.append(example_dict)
        
        # Split into train/val
        random.shuffle(general_examples)
        split_idx = int(0.9 * len(general_examples))
        
        train_examples = general_examples[:split_idx]
        val_examples = general_examples[split_idx:]
        
        # Save datasets
        train_path = os.path.join(self.output_dir, "general", "train.jsonl")
        val_path = os.path.join(self.output_dir, "general", "val.jsonl")
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        
        self._save_jsonl(train_examples, train_path)
        self._save_jsonl(val_examples, val_path)
        
        print(f"General dataset prepared:")
        print(f"  Train: {len(train_examples)} examples")
        print(f"  Val: {len(val_examples)} examples")
        print(f"  Total: {len(general_examples)} examples")
        
        return train_path
    
    def _save_jsonl(self, examples: List[Dict[str, str]], path: str) -> None:
        """Save examples to JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
    
    def create_combined_dataset(self) -> str:
        """
        Create a combined dataset for baseline training.
        
        Returns:
            Path to the combined dataset
        """
        combined_train_examples = []
        combined_val_examples = []
        
        # Load all domain datasets
        domains = ["math", "creative", "general"]
        
        for domain in domains:
            train_path = os.path.join(self.output_dir, domain, "train.jsonl")
            val_path = os.path.join(self.output_dir, domain, "val.jsonl")
            
            if os.path.exists(train_path):
                with open(train_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            example = json.loads(line)
                            combined_train_examples.append(example)
            
            if os.path.exists(val_path):
                with open(val_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            example = json.loads(line)
                            combined_val_examples.append(example)
        
        # Save combined datasets
        combined_dir = os.path.join(self.output_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        
        train_path = os.path.join(combined_dir, "train.jsonl")
        val_path = os.path.join(combined_dir, "val.jsonl")
        
        self._save_jsonl(combined_train_examples, train_path)
        self._save_jsonl(combined_val_examples, val_path)
        
        print(f"Combined dataset created:")
        print(f"  Train examples: {len(combined_train_examples)}")
        print(f"  Val examples: {len(combined_val_examples)}")
        print(f"  Path: {combined_dir}")
        
        return train_path


def create_data_loaders(
    data_path: str,
    tokenizer: SharedTokenizer,
    batch_size: int = 8,  # Reduced for 6GB GPU
    max_length: int = 256,  # Reduced for 6GB GPU
    domain_filter: Optional[str] = None,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_path: Path to the data directory
        tokenizer: Shared tokenizer instance
        batch_size: Batch size for training
        max_length: Maximum sequence length
        domain_filter: Optional domain filter
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = AlchemistDataset(
        os.path.join(data_path, "train.jsonl"),
        tokenizer,
        max_length,
        domain_filter
    )
    
    val_dataset = AlchemistDataset(
        os.path.join(data_path, "val.jsonl"),
        tokenizer,
        max_length,
        domain_filter
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching examples.
    
    Args:
        batch: List of examples
        
    Returns:
        Batched tensors
    """
    # Pad sequences to maximum length in batch
    max_len = max(len(example["input_ids"]) for example in batch)
    
    padded_input_ids = []
    padded_attention_masks = []
    domains = []
    
    for example in batch:
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        
        # Pad to max length
        padding_length = max_len - len(input_ids)
        padded_input_ids.append(
            torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
        )
        padded_attention_masks.append(
            torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
        )
        domains.append(example["domain"])
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "domains": domains
    } 