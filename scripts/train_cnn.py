import argparse
import pickle
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from gensim.models import KeyedVectors
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models.sentence_cnn import SentenceCnn
from utils.text_prepro import build_vocab, load_snips_data, sequence_to_tensor, text_to_indices


def load_embeddings(path, vocab, vocab_size, dim):
    vectors = KeyedVectors.load_word2vec_format(path, binary=True)
    matrix = np.random.uniform(-0.25, 0.25, (vocab_size, dim)).astype(np.float32)
    matrix[0] = 0
    for word, idx in vocab.items():
        cleaned = re.sub('[^0-9a-zA-Z]+', '', word)
        for candidate in (word, word.lower(), cleaned):
            if candidate in vectors:
                matrix[idx] = vectors[candidate]
                break
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/text_cnn_snips.yaml')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])
    torch.manual_seed(cfg['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = cfg['data_files']['SNIPS']

    labels = {}
    train_text, train_y, labels = load_snips_data(paths['train_file'], labels)
    dev_text, dev_y, labels = load_snips_data(paths['dev_file'], labels)

    vocab = build_vocab(train_text, cfg['vocab_size'])
    vocab = {w: i + 4 for w, i in vocab.items()}
    vocab.update({'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3})
    actual_vocab_size = max(vocab.values()) + 1

    model_type = cfg['model_type']
    embeddings = None
    if model_type != 'rand':
        embeddings = load_embeddings(cfg['embedding_file'], vocab, actual_vocab_size, cfg['embedding_dim'])

    pad = int(max(cfg['model_params_cnn']['filter_lengths']) / 2 + 0.5)
    train_x = sequence_to_tensor(text_to_indices(train_text, vocab), (pad, pad))
    dev_x = sequence_to_tensor(text_to_indices(dev_text, vocab), (pad, pad))
    train_y = torch.tensor(train_y)
    dev_y = torch.tensor(dev_y)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg['batch_size'], shuffle=True)
    dev_loader = DataLoader(TensorDataset(dev_x, dev_y), batch_size=cfg['batch_size'])

    model = SentenceCnn(
        len(labels), embeddings, cfg['model_params_cnn']['filter_lengths'],
        cfg['model_params_cnn']['filter_counts'], cfg['dropout_rate'], model_type
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg['optimizer_params']['adam']['lr'],
        weight_decay=cfg['l2_reg_lambda']
    )

    out = Path('runs') / model_type / 'checkpoints'
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(cfg['max_epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in dev_loader:
                pred = model(x.to(device)).argmax(dim=1).cpu()
                correct += int((pred == y).sum())
        acc = correct / len(dev_y)
        print(f'Epoch {epoch + 1:03d} | validation accuracy: {acc * 100:.2f}%')
        if acc > best:
            best = acc
            torch.save({'model_state_dict': model.state_dict(), 'model_type': model_type}, out / 'best.pth')

    with open(out / 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(out / 'labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open(out / 'embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)


if __name__ == '__main__':
    main()
