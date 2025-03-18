"""
Data Config

Specifies the hyperparameters for the dataset, dataloader, and tokenizer.
"""

from dataclasses import dataclass, field

from ._constants import BATCH_SIZE, VOCAB_SIZE


@dataclass
class DatasetConfig:
    # Defines the HuggingFace name of a dataset
    name: str = "pico-lm/pretokenized-dolma"


@dataclass
class DataLoaderConfig:
    # NOTE: You should only change these values jointly with the training config; so that the
    # sub-batch size is consistent with the gradient accumulation steps
    batch_size: int = BATCH_SIZE


@dataclass
class TokenizerConfig:
    # Specify a tokenizer to use
    name: str = "allenai/OLMo-7B-0724-hf"
    vocab_size: int = VOCAB_SIZE


@dataclass
class DataConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
