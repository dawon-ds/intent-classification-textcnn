import numpy as np
import torch
from torch import nn


class ConvFeatures(nn.Module):
    def __init__(self, word_dimension, filter_lengths, filter_counts, dropout_rate):
        super().__init__()
        conv = []
        for size, num in zip(filter_lengths, filter_counts):
            conv2d = nn.Conv2d(1, num, (size, word_dimension))
            nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(conv2d.bias)
            conv.append(nn.Sequential(conv2d, nn.ReLU(inplace=True)))

        self.conv = nn.ModuleList(conv)
        self.filter_sizes = filter_lengths
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embedded_words):
        features = []
        for filter_size, conv in zip(self.filter_sizes, self.conv):
            conv_output = conv(embedded_words)
            conv_output = conv_output.squeeze(-1).max(dim=-1)[0]
            features.append(conv_output)

        features = torch.cat(features, dim=1)
        return self.dropout(features)


class SentenceCnn(nn.Module):
    def __init__(self, nb_classes, word_embedding_numpy, filter_lengths, filter_counts, dropout_rate,
                 model_type='non-static'):
        super().__init__()
        self.model_type = model_type

        if model_type == 'rand':
            vocab_size = 30000
            word_dimension = 300
            self.word_embedding = nn.Embedding(vocab_size, word_dimension, padding_idx=0)
        else:
            if word_embedding_numpy is None:
                raise ValueError("word_embedding_numpy cannot be None for static or non-static model types.")

            word_dimension = word_embedding_numpy.shape[1]
            weights = torch.tensor(word_embedding_numpy.astype(np.float32))

            if model_type == 'static':
                self.word_embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
            elif model_type == 'non-static':
                self.word_embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
            elif model_type == 'multichannel':
                self.word_embedding_static = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
                self.word_embedding_non_static = nn.Embedding.from_pretrained(
                    weights.clone(), freeze=False, padding_idx=0
                )
            else:
                raise ValueError(f"Invalid model_type: {model_type}")

        self.features = ConvFeatures(word_dimension, filter_lengths, filter_counts, dropout_rate)
        self.linear = nn.Linear(sum(filter_counts), nb_classes)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

    def forward(self, input_x):
        if self.model_type == 'multichannel':
            static_embed = self.word_embedding_static(input_x)
            non_static_embed = self.word_embedding_non_static(input_x)
            x = ((static_embed + non_static_embed) / 2).unsqueeze(1)
        else:
            x = self.word_embedding(input_x).unsqueeze(1)

        x = self.features(x)
        return self.linear(x)
