import collections
import os
import re
import unicodedata

import torch


def unicode_to_ascii(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \\( ", text)
    text = re.sub(r"\)", " \\) ", text)
    text = re.sub(r"\?", " \\? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def load_snips_data(folder_path, label_dictionary):
    seq_path = os.path.join(folder_path, 'seq.in')
    label_path = os.path.join(folder_path, 'label')
    with open(seq_path, encoding='utf-8') as f:
        texts = [clean_str(line.strip()) for line in f]
    with open(label_path, encoding='utf-8') as f:
        raw_labels = [line.strip() for line in f]

    labels = []
    for label in raw_labels:
        if label not in label_dictionary:
            label_dictionary[label] = len(label_dictionary)
        labels.append(label_dictionary[label])
    return texts, labels, label_dictionary


def build_vocab(sentences, vocab_size):
    words = []
    for sentence in sentences:
        words.extend(sentence.split())
    vocabulary_inv = [word for word, _ in collections.Counter(words).most_common(vocab_size)]
    return {word: idx for idx, word in enumerate(vocabulary_inv)}


def text_to_indices(texts, word_id_dict, use_unk=True):
    result = []
    for text in texts:
        ids = [2]
        for word in text.split():
            ids.append(
                word_id_dict.get(word, 1)
                if use_unk
                else word_id_dict.setdefault(word, len(word_id_dict))
            )
        ids.append(3)
        result.append(ids)
    return result


def sequence_to_tensor(sequence_list, nb_paddings=(0, 0)):
    front, back = nb_paddings
    max_length = max(map(len, sequence_list)) + front + back
    tensor = torch.zeros((len(sequence_list), max_length), dtype=torch.long)
    for i, sequence in enumerate(sequence_list):
        tensor[i, front:front + len(sequence)] = torch.tensor(sequence)
    return tensor
