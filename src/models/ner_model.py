# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nni.nas.nn.pytorch import ModelSpace
# from nni.nas.nn.pytorch import InputChoice, LayerChoice
# from typing import Dict, Any
# from transformers import AutoModel


# class FixedEmbedding(nn.Module):
#     def __init__(self, embedding_dim: int, num_embeddings: int = 100000):
#         super().__init__()
#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return self.embedding(input)


# class NERModel(ModelSpace):
#     def __init__(self, num_labels: int, config: Dict[str, Any]):
#         super().__init__()
#         self.num_labels = num_labels
#         self.config = config
#         # self.bert=AutoModel.from_pretrained('bert-base-uncased')
#         # for param in self.bert.parameters():
#         #     param.requires_grad = False
#         # Define mutable choices
#         self.model_type = InputChoice(n_candidates=len(config['model_type']['_value']), n_chosen=1, label='model_type')
#         self.embedding_dim = InputChoice(n_candidates=len(config['embedding_dim']['_value']), n_chosen=1, label='embedding_dim')
#         self.rnn_hidden_size = InputChoice(n_candidates=len(config['rnn_hidden_size']['_value']), n_chosen=1, label='rnn_hidden_size')
#         self.rnn_num_layers = InputChoice(n_candidates=len(config['rnn_num_layers']['_value']), n_chosen=1, label='rnn_num_layers')
#         self.rnn_dropout = InputChoice(n_candidates=len(config['rnn_dropout']['_value']), n_chosen=1, label='rnn_dropout')

#         # Embedding layer
#         self.word_embeddings = FixedEmbedding(max(config['embedding_dim']['_value']))

#         # LayerChoice for RNN and CNN+BiLSTM selection
#         self.rnn_layer_choice = LayerChoice([
#             nn.LSTM(max(config['embedding_dim']['_value']),
#                     max(config['rnn_hidden_size']['_value']),
#                     num_layers=max(config['rnn_num_layers']['_value']),
#                     bidirectional=True, batch_first=True, dropout=max(config['rnn_dropout']['_value'])),
#             nn.GRU(max(config['embedding_dim']['_value']),
#                    max(config['rnn_hidden_size']['_value']),
#                    num_layers=max(config['rnn_num_layers']['_value']),
#                    bidirectional=True, batch_first=True, dropout=max(config['rnn_dropout']['_value']))
#         ], label='rnn_layer')

#         # Transformer layers
#         self.transformer_layer_choice = LayerChoice([
#             nn.TransformerEncoder(
#                 nn.TransformerEncoderLayer(
#                     d_model=max(config['transformer_d_model']['_value']),
#                     nhead=max(config['transformer_num_heads']['_value']),
#                     dim_feedforward=max(config['transformer_dim_feedforward']['_value']),
#                     dropout=max(config['transformer_dropout']['_value']),
#                     activation=max(config['transformer_activation']['_value'])
#                 ),
#                 num_layers=max(config['transformer_num_layers']['_value'])
#             )
#         ], label='transformer_layer')

#         # CNN Layer (for CNN + BiLSTM)
#         self.cnn_layer_choice = LayerChoice([
#             nn.Conv2d(1, max(config['cnn_num_filters']['_value']), kernel_size=(ks, max(config['embedding_dim']['_value'])))
#             for ks in config['cnn_kernel_sizes']['_value'][0]
#         ], label='cnn_layer')

#         self.hidden2tag = nn.Linear(max(config['rnn_hidden_size']['_value']) * 2, num_labels)

#     def forward(self, word_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         model_type = self.config['model_type']['_value'][0]

#         # Embedding layer
#         # outputs = self.bert(input_ids=word_ids, attention_mask=attention_mask)
#         # embedded = outputs.last_hidden_state 
#         # print(embedded.shape)

#         embedded=self.word_embeddings(word_ids)

#         if model_type == 'BiLSTM':
#             rnn = self.rnn_layer_choice
#             output, _ = rnn(embedded)
#         elif model_type == 'Transformer':
#             transformer = self.transformer_layer_choice
#             output = transformer(embedded)
#         elif model_type == 'CNN_BiLSTM':
#             embedded = embedded.unsqueeze(1) 
#             cnn = self.cnn_layer_choice
#             cnn_out = cnn(embedded).squeeze(3)  # Squeeze to remove channel dimension
#             output, _ = self.rnn_layer_choice(cnn_out.permute(0, 2, 1)) 
#         else:
#             raise ValueError(f"Unsupported model type: {model_type}")

#         logits = self.hidden2tag(output)
#         return logits

# def get_model_space(num_labels: int, search_space: Dict[str, Any]) -> NERModel:
#     return NERModel(num_labels, search_space)

# def create_model(num_labels: int, search_space: Dict[str, Any]) -> nn.Module:
#     return NERModel(num_labels, search_space)
import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.nas.nn.pytorch import ModelSpace
from nni.nas.nn.pytorch import InputChoice, LayerChoice
from typing import Dict, Any
from torchcrf import CRF

class FixedEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int = 100000):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)

class Highway(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)

    def forward(self, x):
        transform = F.relu(self.transform(x))
        gate = torch.sigmoid(self.gate(x))
        return gate * transform + (1 - gate) * x

class BLSTM_CNN_CRF(ModelSpace):
    def __init__(self, num_labels: int, config: Dict[str, Any]):
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        # Embedding layer
        self.embedding_dim = InputChoice(n_candidates=len(config['embedding_dim']['_value']), n_chosen=1, label='embedding_dim')
        self.word_embeddings = FixedEmbedding(max(config['embedding_dim']['_value']))

        # CNN layer
        self.cnn_filters = InputChoice(n_candidates=len(config['cnn_filters']['_value']), n_chosen=1, label='cnn_filters')
        self.cnn_kernel_size = InputChoice(n_candidates=len(config['cnn_kernel_size']['_value']), n_chosen=1, label='cnn_kernel_size')
        self.cnn = nn.Conv1d(
            max(config['embedding_dim']['_value']),
            max(config['cnn_filters']['_value']),
            kernel_size=max(config['cnn_kernel_size']['_value']),
            padding=max(config['cnn_kernel_size']['_value']) // 2
        )

        # LSTM layer
        self.lstm_hidden_size = InputChoice(n_candidates=len(config['lstm_hidden_size']['_value']), n_chosen=1, label='lstm_hidden_size')
        self.lstm_num_layers = InputChoice(n_candidates=len(config['lstm_num_layers']['_value']), n_chosen=1, label='lstm_num_layers')
        self.lstm_dropout = InputChoice(n_candidates=len(config['lstm_dropout']['_value']), n_chosen=1, label='lstm_dropout')
        self.lstm = nn.LSTM(
            max(config['cnn_filters']['_value']),
            max(config['lstm_hidden_size']['_value']),
            num_layers=max(config['lstm_num_layers']['_value']),
            bidirectional=True,
            batch_first=True,
            dropout=max(config['lstm_dropout']['_value'])
        )

        # Highway network
        self.use_highway = InputChoice(n_candidates=2, n_chosen=1, label='use_highway')
        self.highway = Highway(max(config['lstm_hidden_size']['_value']) * 2)

        # Linear layer
        self.hidden2tag = nn.Linear(max(config['lstm_hidden_size']['_value']) * 2, num_labels)

        # CRF layer
        self.use_crf = InputChoice(n_candidates=2, n_chosen=1, label='use_crf')
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, word_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Word embedding
        embedded = self.word_embeddings(word_ids)
        
        # CNN
        cnn_out = self.cnn(embedded.transpose(1, 2)).transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)
        
        # Highway network (optional)
        if self.use_highway.chosen == 0:
            lstm_out = self.highway(lstm_out)
        
        # Linear
        emissions = self.hidden2tag(lstm_out)
        
        return emissions

    import torch.nn.functional as F

    def loss(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor, class_weights: torch.Tensor = None) -> torch.Tensor:
        if self.use_crf.chosen == 0:
            # CRF loss (no direct class weighting support)
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        else:
            # Cross-entropy loss with class weights
            return F.cross_entropy(emissions.view(-1, self.num_labels), tags.view(-1), weight=class_weights, ignore_index=-100)


    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_crf.chosen == 0:
            return self.crf.decode(emissions, mask=mask)
        else:
            return emissions.argmax(dim=-1)

def get_model_space(num_labels: int, search_space: Dict[str, Any]) -> BLSTM_CNN_CRF:
    return BLSTM_CNN_CRF(num_labels, search_space)

def create_model(num_labels: int, search_space: Dict[str, Any]) -> nn.Module:
    return BLSTM_CNN_CRF(num_labels, search_space)