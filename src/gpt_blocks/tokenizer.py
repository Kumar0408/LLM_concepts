import re

class SimpleTokenizerV1:
    """Simple tokenizer that splits text into words and 
    generates token ID based on the vocabulary provided.
    """
    def __init__(self, vocab:dict):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()} # reverse mapping
        self.preprocessed_text =[]
        self.ids = []

    def encode(self, text:str)->list:
        """Convert text to token IDs."""

        self.preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        self.preprocessed_text = [
            item.strip() for item in self.preprocessed_text if item.strip()
        ]
        self.ids = [self.str_to_int[s] for s in self.preprocessed_text]
        return self.ids
    
    def decode(self, ids:list[int])->str:
        """Convert token IDs to text."""

        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Replace unknown words with <|unk|>
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text