# Tokenization

- Commonly vocabulary sizes is around 30k to 60k tokens in 2022.

## Handling Unknown words

- Solution 1
  - Use `<UNK>` token
  - We do this by replacing the low-frequency words in our corpus with `<UNK>` and then use it for uknown words at test time.
  - Issues: We are collapsing a lot of information into 1 token, which can be confusing.

- Solution 2

    - Character Level Tokenization.
    - Your vocabulary is now the set of all characters in your corpus.
    - This results in longer input sequences

----------

## Subword Tokenization

- A midpoint between Character level and Word level tokenization.
- Examples of this include BPE, WordPiece, and Unigram Language Models.

- You can't make sure that the model doesn't output a subword that doesn't make sense `ed` for example, but with enough training, this might happen only due to sampling.

The vocabulary contains the all the steps of the tokenization process.

----------

### SentecePiece

- SentencePiece can train subowrd models directly from **raw sentences**, which allows to make a **purely end-to-end** and **language independent system**.
- Most tokenizers are designed for **European languages** where words are **segmented with white spaces**.
- SentecePiece implements two subword segmentation algorithms, **BPE & Unigram language model** with the extension of **direct training from raw sentences**

#### Components

- SentencePiece consists of 4 components.
  - Normalizer: Normalize semantically equivalent unicode characters into **canonical forms**
  - Trainer: Trains a subword model from normalizer sentences, We specify a type of subword model as the parameter of the trainer.
  - Encoder: Normalizes and encodes sentences into subword ids.
  - Decoder: Decodes subword ids into sentences.

----------

## Tokenizer Free Model: ByT5


