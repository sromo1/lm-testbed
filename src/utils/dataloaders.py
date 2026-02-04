import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt:str, tokenizer, max_length:int, stride:int):
        """
        Docstring for __init__
        
        :param txt: Text to load to the dataset
        :type txt: str
        :param tokenizer: Tokenizer
        :param max_length: Sliding window length
        :type max_length: int
        :param stride: Sliding windows 'steps' to control overlap between dataset samples
        :type stride: int
        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids)-max_length, stride):       # Sliding window to chunk into overlapping sequences of max_length
            input_chunk = token_ids[i:i+max_length]
            output_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))

    def __getitem__(self, index):
        """ 
        Retrieves exactly one data record and corresponding label 
        """
        return self.input_ids[index], self.target_ids[index]

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.input_ids)

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                       # Initialize tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)      # Create dataset
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )
    return dataloader