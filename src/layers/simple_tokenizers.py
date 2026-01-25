import re

class SimpleTokenizerV1:
    def __init__(self, vocab:dict[str:int]):
        self.str_to_int:dict[str,int] = vocab
        self.int_to_str:dict[int,str] = {i:s for s,i in vocab.items()}

    def encode(self, text:str) -> list[int]:
        preprocessed = re.split(r'([,.?_!\"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[i] for i in preprocessed]
        return ids
    
    def decode(self, ids:list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?_!\"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    """Include <|unk|> and <|endoftext|> tokens"""
    def __init__(self, vocab:dict[str,int]):
        self.str_to_int:dict[str,int] = vocab
        self.int_to_str:dict[int,str] = {i:s for s,i in vocab.items()}

    def encode(self, text:str) -> list[int]:
        preprocessed = re.split(r'([,.?_!\"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[i] for i in preprocessed]
        return ids
    
    def decode(self, ids:list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?_!\"()\'])', r'\1', text)
        return text
