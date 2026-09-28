import argparse
import pickle
from pathlib import Path

import torch
import yaml

from models.sentence_cnn import SentenceCnn
from utils.text_prepro import clean_str, sequence_to_tensor, text_to_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text')
    parser.add_argument('--config', default='config/text_cnn_snips.yaml')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = Path('runs') / cfg['model_type'] / 'checkpoints'

    with open(ckpt_dir / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open(ckpt_dir / 'labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(ckpt_dir / 'embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    pad = int(max(cfg['model_params_cnn']['filter_lengths']) / 2 + 0.5)
    x = sequence_to_tensor(text_to_indices([clean_str(args.text)], vocab), (pad, pad)).to(device)

    model = SentenceCnn(
        len(labels), embeddings, cfg['model_params_cnn']['filter_lengths'],
        cfg['model_params_cnn']['filter_counts'], cfg['dropout_rate'], cfg['model_type']
    ).to(device)
    state = torch.load(ckpt_dir / 'best.pth', map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    with torch.no_grad():
        idx = int(model(x).argmax(dim=1).item())

    inverse = {v: k for k, v in labels.items()}
    print(inverse[idx])


if __name__ == '__main__':
    main()
