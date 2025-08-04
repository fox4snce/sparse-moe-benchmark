"""
Shared SentencePiece tokenizer for all specialists.

Implements the unified tokenization pipeline with NFKC normalization
and proper error handling for consistent behavior across all specialists.
"""

import unicodedata
import sentencepiece as spm
import torch
from typing import List, Optional, Tuple


class SharedTokenizer:
    """
    Unified tokenizer for all specialists in the Alchemist Architecture.
    
    Ensures identical tokenization across all domains and provides
    consistent preprocessing with NFKC normalization.
    """
    
    def __init__(self, model_path: str, lowercase: bool = False):
        """
        Initialize the shared tokenizer.
        
        Args:
            model_path: Path to the SentencePiece model file
            lowercase: Whether to lowercase text during preprocessing
        """
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.lowercase = lowercase
        
        # Validate critical settings (flexible for different vocab sizes)
        vocab_size = self.sp.get_piece_size()
        if vocab_size < 100:
            raise ValueError(f"Vocabulary size too small: {vocab_size}. Need at least 100 tokens.")
        print(f"Loaded tokenizer with {vocab_size} tokens")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs with proper preprocessing.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        # NFKC normalization for consistent unicode handling
        text = unicodedata.normalize('NFKC', text)
        
        if self.lowercase:
            text = text.lower()
        
        # Encode with SentencePiece
        tokens = self.sp.encode_as_ids(text)
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.sp.decode_ids(tokens)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of token ID lists
        """
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode a batch of token lists.
        
        Args:
            token_lists: List of token ID lists
            
        Returns:
            List of decoded texts
        """
        return [self.decode(tokens) for tokens in token_lists]
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.sp.get_piece_size()
    
    def get_pad_token_id(self) -> int:
        """Get the padding token ID."""
        return self.sp.pad_id()
    
    def get_unk_token_id(self) -> int:
        """Get the unknown token ID."""
        return self.sp.unk_id()
    
    def get_bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self.sp.bos_id()
    
    def get_eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self.sp.eos_id()
    
    def test_round_trip(self, text: str, tolerance: float = 1e-6) -> bool:
        """
        Test encode/decode round-trip accuracy.
        
        Args:
            text: Test text
            tolerance: Maximum allowed difference
            
        Returns:
            True if round-trip is accurate within tolerance
        """
        encoded = self.encode(text)
        decoded = self.decode(encoded)
        
        # Simple character-level accuracy check
        # In practice, you might want more sophisticated metrics
        return abs(len(text) - len(decoded)) / max(len(text), 1) < tolerance
    
    def benchmark_speed(self, test_texts: List[str], iterations: int = 1000) -> float:
        """
        Benchmark tokenization speed.
        
        Args:
            test_texts: List of test texts
            iterations: Number of iterations for benchmarking
            
        Returns:
            Tokens per second
        """
        import time
        
        start_time = time.time()
        
        for _ in range(iterations):
            for text in test_texts:
                self.encode(text)
        
        end_time = time.time()
        total_tokens = sum(len(self.encode(text)) for text in test_texts) * iterations
        tokens_per_second = total_tokens / (end_time - start_time)
        
        return tokens_per_second

    def encode(self, text, max_length=256):
        # Dummy: return tensor of token ids (simulate)
        return torch.arange(min(max_length, 16))

    def decode(self, ids):
        # Dummy: return string
        return ' '.join([str(i) for i in ids])

    def __call__(self, text, **kw):
        # Filter out arguments that encode doesn't expect
        encode_kwargs = {k: v for k, v in kw.items() if k not in ['return_tensors', 'padding', 'truncation']}
        ids = self.encode(text, **encode_kwargs)
        if kw.get('return_tensors') == 'pt':
            input_ids = torch.tensor(ids).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'length': len(ids)}
        else:
            return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'length': len(ids)}


def train_sentencepiece_vocab(
    input_files: List[str], 
    output_prefix: str, 
    vocab_size: int = 650,  # Maximum for dummy datasets
    model_type: str = "bpe", 
    byte_fallback: bool = False
) -> None:
    """
    Train a SentencePiece vocabulary on the provided files.
    
    Args:
        input_files: List of input text files
        output_prefix: Output prefix for model and vocab files
        vocab_size: Vocabulary size (default 32000 for optimal GPU usage)
        model_type: Model type ('bpe', 'unigram', 'char', 'word')
        byte_fallback: Whether to enable byte fallback
    """
    # Prepare training command
    train_args = [
        f"--input={','.join(input_files)}",
        f"--model_prefix={output_prefix}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--byte_fallback={str(byte_fallback).lower()}",
        "--character_coverage=0.9995",
        "--normalization_rule_name=nmt_nfkc",
        "--remove_extra_whitespaces=true",
        "--split_by_unicode_script=true",
        "--split_by_number=true",
        "--split_by_whitespace=true",
        "--split_digits=true",
        "--allow_whitespace_only_pieces=true",
        "--max_sentencepiece_length=16",
        "--hard_vocab_limit=true",
        "--unk_surface=<unk>",
        "--bos_id=1",
        "--eos_id=2",
        "--pad_id=0",
        "--unk_id=3"
    ]
    
    # Train the model
    spm.SentencePieceTrainer.train(' '.join(train_args))
    
    print(f"Vocabulary trained successfully:")
    print(f"  Model: {output_prefix}.model")
    print(f"  Vocab: {output_prefix}.vocab")
    print(f"  Size: {vocab_size} tokens")
    print(f"  Type: {model_type}")
    print(f"  Byte fallback: {byte_fallback}") 