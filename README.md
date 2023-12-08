#  Probing the impact from fine-tuning on BERT's ability in understanding toxic speech detection

This section of the repo is dedicated to the final project for the course 
W266: Natural Language Processing with Deep Learning for the term of Fall 2023.
The final paper can be found under the `report` directory.

## Abstract

This paper investigates the ability of the contextual representations learned and generated by BERT, a Transformer-based pre-trained language model, in understanding toxic speech detection before and after fine-tuning. More concretely, we aim to shed light on the following question: How does the linguistic knowledge captured and encoded by the contextual embeddings during pre-training and fine-tuning impact the performance of the downstream task of toxic speech detection? We conduct our experiments using probing tasks, which are supervised classification tasks designed to analyze the linguistic knowledge captured and encoded in contextual embeddings, based on a simple linear probing model architecture and task datasets on topics often associated with toxic speech. We find that pre-trained contextual embeddings already have the capacity to identify on- and off-topic mentions in the presence and absence of toxicity. Our analysis reveals that the fine-tuned contextual embeddings do not have the ability to distinguish individual toxic components of a post, which suggests that toxicity of individual words is not encoded in the contextual embeddings even after fine-tuning. We also observe that fine-tuning yields varying performance impact on contextual embeddings across all hidden layers where some of the best performing contextual embeddings are found in earlier hidden layers deep in the Transformer stack of a BERT model.

:warning: **Warning: This paper contains samples of texts that are offensive and/or hateful in nature.** :warning:

## Organization

The following illustrates the organization of the directory as used by the code.
Due to size limits, we do not share any data used or generated as part of the
experiments. For the original hateXplain dataset, please refer to [this repo](https://github.com/hate-alert/HateXplain/tree/master).

```
├── README.md
├── data
│   ├── hatexplain          <- Datasets from hateXplain
│   ├── hidden_states       <- Hidden states as extracted from finetuned models
│   ├── probe_results       <- Results from probing experiments
│   └── token_maps          <- Mappings from post tokens to BERT subword tokens
├── models
│   └── model_checkpoints   <- Saved model weights
├── poetry.lock
├── pyproject.toml          <- Poetry environment configuration
├── report                  <- Final report
└── src
    ├── main.py             <- Sample code on using the utils
    └── utils               
        ├── __init__.py
        ├── dataset.py      <- Utilities for managing datasets
        ├── demo.py         <- Utilities for demo purposes
        ├── model.py        <- Utilities for managing models
        ├── probe.py        <- Utilities for managing probes
        └── token_map.py    <- Utilities for managing token maps
```

## Usage

Please refer to `/src/main.py` for examples on how the utilities are used in
our experiments.