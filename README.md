# Intent Classification with Text CNN

Text CNN experiments for **7-class intent classification on the SNIPS dataset**, comparing different word-embedding strategies under the same CNN architecture.

## Project Overview

The goal of this project was to examine how the choice of word-embedding strategy affects intent-classification performance. Four variants were evaluated: randomly initialized embeddings, frozen pretrained embeddings, trainable pretrained embeddings, and a combined frozen/trainable embedding variant.

## Model

The classifier uses convolution filters with window sizes **3, 4, and 5**, with **100 filters per size**, followed by ReLU, max-over-time pooling, dropout, and a fully connected classification layer.

Pretrained variants use **300-dimensional Word2Vec embeddings**. The experiment used Adam with a learning rate of `0.001`, dropout of `0.5`, L2 regularization of `1e-4`, and a batch size of `64`.

> **Implementation note:** In the submitted code, the `multichannel` variant creates a frozen and a trainable embedding table, averages the two embedding outputs, and then feeds the result into a shared CNN feature extractor. Therefore, it is described here according to the actual implementation rather than as a conventional two-channel CNN with separate convolution paths.

## Results

| Model | Test Accuracy |
|---|---:|
| CNN-rand | 96.55% |
| CNN-non-static | 96.71% |
| CNN-static | 97.71% |
| CNN-multichannel | **98.14%** |

![Model comparison](results/model_comparison.png)

The combined frozen/trainable embedding variant achieved the highest recorded test accuracy in the submitted evaluation report.

## Project Structure

```text
intent-classification-textcnn/
├── config/
│   └── text_cnn_snips.yaml
├── data/
│   └── README.md
├── models/
│   └── sentence_cnn.py
├── results/
│   └── model_comparison.png
├── scripts/
│   ├── train_cnn.py
│   ├── eval_cnn.py
│   └── infer_cnn.py
├── utils/
│   └── text_prepro.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Notes

The SNIPS dataset and Google News Word2Vec binary used during the original experiment are not included in this repository. Local machine-specific paths from the coursework submission were removed for portfolio use. See `data/README.md` for the expected directory structure.
