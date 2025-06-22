from functools import cached_property

import torch
from transformers import AutoModel


class HFEmbeddingsExtractor:
    """Extracts embeddings from a Hugging Face model."""

    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path
    
    @cached_property
    def embeddings(self):
        model = AutoModel.from_pretrained(self.name_or_path)
        model.eval()
        embeddings = model.embed_tokens
        return embeddings
    
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name_or_path}')"

    def extract(self, token_ids):
        with torch.no_grad():
            token_ids = torch.tensor(token_ids)
            return self.embeddings.forward(token_ids).squeeze().numpy()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        if hasattr(self, 'embeddings'):
            del self.embeddings