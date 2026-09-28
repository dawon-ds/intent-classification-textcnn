import argparse
import pickle
from pathlib import Path

import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from models.sentence_cnn import SentenceCnn
from utils.text_prepro import load_snips_data, sequence_to_tensor, text_to_indices


def main():
    parser = argparse.ArgumentParser()
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

    test_text, test_y, _ = load_snips_data(cfg['data_files']['SNIPS']['test_file'], labels.copy())
    pad = int(max(cfg['model_params_cnn']['filter_lengths']) / 2 + 0.5)
    test_x = sequence_to_tensor(text_to_indices(test_text, vocab), (pad, pad))
    test_y = torch.tensor(test_y)
    loader = DataLoader(TensorDataset(test_x, test_y), batch_size=cfg['batch_size'])

    model = SentenceCnn(
        len(labels), embeddings, cfg['model_params_cnn']['filter_lengths'],
        cfg['model_params_cnn']['filter_counts'], cfg['dropout_rate'], cfg['model_type']
    ).to(device)
    checkpoint = torch.load(ckpt_dir / 'best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.extend(model(x.to(device)).argmax(dim=1).cpu().tolist())

    accuracy = sum(a == b for a, b in zip(preds, test_y.tolist())) / len(test_y)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    names = [name for name, _ in sorted(labels.items(), key=lambda kv: kv[1])]
    print(classification_report(test_y.tolist(), preds, target_names=names, digits=4))


if __name__ == '__main__':
    main()
