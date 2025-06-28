import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging

class SimpleMedicalDataset(Dataset):
    def __init__(self, pickle_file, tokenizer, max_length=8192, use_clinical_features=True):
        """Simplified dataset loader for our medical text classification :)"""
        self.df = pd.read_pickle(pickle_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_clinical_features = use_clinical_features
        
        print(f"Loaded dataset with {len(self.df)} samples")
        print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")
        print(f"Text length stats: min={self.df['text_length'].min()}, "
              f"max={self.df['text_length'].max()}, mean={self.df['text_length'].mean():.1f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = int(self.df.iloc[idx]['label'])
        
        # Implement chunking for very long texts
        if len(text) > 200000:  # Character count threshold
            # Process in chunks to avoid OOM during tokenization
            chunk_size = 100000  # Characters
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            all_input_ids = []
            for chunk in chunks:
                chunk_tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
                all_input_ids.extend(chunk_tokens)

            # Trim to max_length and add special tokens
            all_input_ids = all_input_ids[:self.max_length-2]  # Allow room for special tokens
            all_input_ids = [self.tokenizer.bos_token_id] + all_input_ids + [self.tokenizer.eos_token_id]

            # Pad to max_length
            attention_mask = [1] * len(all_input_ids)
            padding_length = self.max_length - len(all_input_ids)
            all_input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

            encodings = {
                'input_ids': torch.tensor(all_input_ids),
                'attention_mask': torch.tensor(attention_mask)
            }
        else:
            # Standard tokenization for shorter texts
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        sample = {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'label': torch.tensor(label),
        }
        
        # Add clinical features if requested (checking to see if they will have an effect)
        if self.use_clinical_features:
            # Convert survival to float tensor
            if 'overallsurvival' in self.df.columns:
                sample['overallsurvival'] = torch.tensor(float(self.df.iloc[idx]['overallsurvival']))
            
            # Convert stage_grade to one-hot encoding
            if 'stage_grade' in self.df.columns:
                stage = self.df.iloc[idx]['stage_grade']
                if stage == '0-2':
                    stage_tensor = torch.tensor([1.0, 0.0, 0.0])
                elif stage == '3':
                    stage_tensor = torch.tensor([0.0, 1.0, 0.0])
                elif stage == '4':
                    stage_tensor = torch.tensor([0.0, 0.0, 1.0])
                else:
                    stage_tensor = torch.tensor([0.0, 0.0, 0.0])
                sample['stage_grade'] = stage_tensor
        
        return sample