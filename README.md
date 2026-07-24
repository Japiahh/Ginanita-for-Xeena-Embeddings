<img height="35" src="https://i.imgflip.com/adfqd6.gif"/> # Ginanita-for-Xeena-Embeddings <img height="35" src="https://i.imgflip.com/adfqd6.gif"/>
Embeddings modul for Xeena : Linguistic Aware Transformers

**Ginanita** is a custom *Multi-Feature Linguistic Injector* designed to be the input embedding layer for Indonesian Language Models.

Instead of just relying on the usual Token Embedding and Positional Embedding you see in standard Transformers, Ginanita injects 6 different linguistic features all at once right into the initial sentence representation.

## The Linguistic Specialists

Ginanita orchestrates a bunch of embedding "specialists." Each one is named after a character, giving the project a bit of its own personality:

1. **Adelina** (Token Embedding): Turns word tokens into base vectors.
2. **Raya** (Positional Embedding): Encodes word order using sine waves (fixed position).
3. **Elina** (Morphology/POS Embedding): Grabs info about word classes (like Verbs, Nouns, etc).
4. **Lekha Syntactic**: Injects syntax tree labels (NP, VP) along with their tree depth.
5. **Lekha Dependency**: Captures dependency relations between words (like root, subject, object).
6. **Sentence ID**: Keeps track of whether a word belongs to the first sentence, second sentence, and so on in a sequence.

## Installation

Just clone this repo and install it via pip:

```bash
git clone https://github.com/username/Ginanita.git
cd Ginanita
pip install -e .
```

## Quick Start

Here's a quick example of how you can set up and use Ginanita in your project:

```python
import torch
from ginanita import Ginanita, GinanitaConfig

# 1. Setup Config
config = GinanitaConfig(
    vocab_size=10000,
    d_model=512,
    block_size=256,
    feature_dropout=0.1  # Turn on feature dropout during training
)

# 2. Init the Model
injector = Ginanita(config)

# 3. Create some dummy inputs (Batch=2, SeqLen=10)
B, T = 2, 10
inputs = {
    'idx': torch.randint(0, config.vocab_size, (B, T)),
    'pos_ids': torch.randint(0, config.pos_size, (B, T)),
    'syn_ids': torch.randint(0, config.syn_size, (B, T)),
    'depth_ids': torch.randint(0, config.max_depth, (B, T)),
    'dep_ids': torch.randint(0, config.dep_size, (B, T)),
    'sent_ids': torch.zeros(B, T, dtype=torch.long)
}

# 4. Forward Pass
output = injector(**inputs)
print("Output shape:", output.shape)  
```

## Key Features

- **Feature Dropout**: A special mechanism used during the *training* phase where supporting features (Morphology, Syntax, Dependency) can be randomly "turned off" independently. This forces the model to not rely too heavily on any single feature (the main Token feature is always kept safe).
- **Concatenation & Projection**: Combines all features into one high-dimensional space and then projects them back down to `d_model` size using a Linear Projection layer.

Start to Risa, thanks Gemie.
