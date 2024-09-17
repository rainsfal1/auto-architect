import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoConfig


class BaseNERModel(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_dim=100, hidden_size=256, num_layers=2, dropout=0.5,
                 use_crf=True, use_bert=False):
        super(BaseNERModel, self).__init__()

        self.num_labels = num_labels
        self.use_crf = use_crf
        self.use_bert = use_bert

        if use_bert:
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            self.embedding_dim = self.bert.config.hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_labels)

        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.use_bert:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = self.embedding(input_ids)

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        emissions = self.fc(lstm_output)

        if self.use_crf:
            if labels is not None:
                loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
                return loss
            else:
                return self.crf.decode(emissions, mask=attention_mask.byte())
        else:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = emissions.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                return loss
            else:
                return torch.argmax(emissions, dim=2)


def create_base_model(num_labels, vocab_size, config=None):
    if config is None:
        config = {
            'embedding_dim': 100,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.5,
            'use_crf': True,
            'use_bert': False
        }

    return BaseNERModel(
        num_labels=num_labels,
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_crf=config['use_crf'],
        use_bert=config['use_bert']
    )


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'embedding_dim': 100,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.5,
        'use_crf': True,
        'use_bert': False
    }

    # Create model
    num_labels = 9  # Example: number of NER tags
    vocab_size = 10000  # Example: size of your vocabulary
    model = create_base_model(num_labels, vocab_size, config)

    # Example input
    batch_size = 32
    seq_length = 50
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    labels = torch.randint(0, num_labels, (batch_size, seq_length))

    # Forward pass
    loss = model(input_ids, attention_mask, labels)
    print(f"Loss: {loss.item()}")

    # Inference
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
    print(f"Predictions shape: {predictions.shape}")
