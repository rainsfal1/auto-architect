from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import load_dataset, load_from_disk
import os
import torch

# Global variables to store the data loaders and class weights
_train_loader = None
_val_loader = None
_test_loader = None
_weights = None

def create_data_loaders(batch_size=32, dataset_path='encoded_conll2003_dataset'):
    global _train_loader, _val_loader, _test_loader, _weights
    
    if _train_loader is None and _val_loader is None and _test_loader is None:
        # Check if the encoded dataset exists
        if not os.path.exists(dataset_path):
            # Load the dataset
            dataset = load_dataset("conll2003")
            
            # Initialize the tokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            def tokenize_and_align_labels(examples):
                tokenized_inputs = tokenizer(
                    examples['tokens'],
                    truncation=True,
                    padding='longest',
                    is_split_into_words=True
                )

                labels = []
                for i, label in enumerate(examples['ner_tags']):
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    previous_word_idx = None
                    label_ids = []

                    for word_idx in word_ids:
                        if word_idx is None:
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)

                        previous_word_idx = word_idx

                    labels.append(label_ids)

                tokenized_inputs['labels'] = labels
                return tokenized_inputs

            # Tokenize and align labels
            encoded_dataset = dataset.map(tokenize_and_align_labels, batched=True)
            
            # Save the encoded dataset
            encoded_dataset.save_to_disk(dataset_path)
        else:
            # Load the encoded dataset
            encoded_dataset = load_from_disk(dataset_path)

        # Set format and remove unnecessary columns
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        encoded_dataset = encoded_dataset.remove_columns(['tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])

        # Initialize the tokenizer and data collator
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

        # Create DataLoaders
        _train_loader = DataLoader(encoded_dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        _val_loader = DataLoader(encoded_dataset['validation'], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        _test_loader = DataLoader(encoded_dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=data_collator)

        # Calculate class weights based on the training data
        _weights = calculate_class_weights(_train_loader)

def calculate_class_weights(train_loader):
    class_counts = torch.zeros(9)
    for batch in train_loader:
        labels = batch['labels']
        for label in labels.flatten():
            if label != -100:  # Ignore padding
                class_counts[label] += 1

    total_samples = class_counts.sum()
    class_weights = total_samples / (9 * class_counts)
    class_weights[class_counts == 0] = 0  # Avoid division by zero
    return class_weights

def get_data_loaders():
    if _train_loader is None or _val_loader is None or _test_loader is None:
        raise ValueError("Data loaders have not been created. Please call `create_data_loaders` first.")
    return _train_loader, _val_loader, _test_loader

def get_weights():
    if _weights is None:
        raise ValueError("Class weights have not been calculated. Please call `create_data_loaders` first.")
    return _weights
