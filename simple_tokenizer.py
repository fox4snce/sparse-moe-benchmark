import re
import torch

class SimpleTokenizer:
    """
    Simple, reproducible tokenizer: whitespace + punctuation split, lowercased.
    No external files, no dependencies except torch and re.
    Suitable for benchmarks, unit tests, and rapid prototyping.
    Not for production LLMs—token count is higher than BPE/WordPiece.
    """
    def __init__(self, vocab_size=650):
        self.vocab_size = vocab_size
        self.unk_token_id = 1
        self.pad_token_id = 0

    def tokenize(self, text):
        # Lowercase, split on whitespace, then split out punctuation
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def encode(self, text, return_tensors=None):
        if isinstance(text, list):
            # Batch mode
            all_ids = [self.encode(t) for t in text]
            if return_tensors == "pt":
                maxlen = max(len(ids) for ids in all_ids)
                padded = [ids + [self.pad_token_id]*(maxlen-len(ids)) for ids in all_ids]
                return torch.tensor(padded, dtype=torch.long)
            return all_ids
        tokens = self.tokenize(text)
        # Assign IDs (for demo, hash to fit in vocab)
        ids = [abs(hash(tok)) % self.vocab_size for tok in tokens]
        if return_tensors == "pt":
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # For demo, just return dummy tokens
        return " ".join([f"token_{i}" for i in ids])

    def __call__(self, text, **kwargs):
        ids = self.encode(text, return_tensors="pt")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return {
            "input_ids": ids,
            "attention_mask": (ids != self.pad_token_id).long()
        }