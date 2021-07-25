# <u>**HuggingFaceNotes**</u>

# <u>**Chapter 1**</u>

## <u>Types of Transformers</u>

- **<u>Encoder</u>**

  - Only uses the encoder part of the transformer.
  - Used when you need to understand the contents of the sentence like **sequence classification**.

  - Examples Include: **BERT, RoBERTa, AlBERT & DistilBERT**

- **<u>Decoder</u>**

  - Only uses the decoder part of the transformer and are also called **auto-regressive** models.

  - Best suited for **text generation** as they revolve around predicting the **next word in a sentence**

  - Examples include: **GPT, GPT-2, CTRL & Transformer XL**

- **<u>Seq2Seq</u>**

  - Use both the encoder and decoder parts of the transformer.
  - Suitable for tasks revolving around **generating new sentences** depending on a **given input** (**conditional text generation**)
  - Tasks include: **Machine Translation**, **Summarization** and **generative question answering**
  - Examples include: **BART, mBART, Marian & T5**

****

# **<u>Chapter 2</u>**

## **<u>Inside the pipeline</u>**

Inside a HuggingFace pipeline there are three main stages

- **<u>Tokenizer</u>**
- **<u>Model</u>**
- **<u>Post-Processing</u>**

![](./Images/Chapter2/full_nlp_pipeline.png)

****

### **<u>Tokenizer</u>**

Since each different architecture was pretrained in a certain way, each one has its own tokenizer with its own **special tokens** like **start tokens**, **Mask tokens** .... etc.

In huggingface, the job of the tokenizer is to:

- Splitting the input into words, subwords, or symbols (like punctuation) that are called *tokens*
- Mapping each token to an integer
- Adding additional inputs that may be useful to the model

****

#### <u>**Importing DistilBERT**</u>

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

We can now use our tokenizer to tokenize sentences in the `DistilBERT` way.

```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

The `return_tensors='pt'` tells the tokenizer to return output as a pyTorch `Tensor`. The default return is a python `list`

```python
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

`attention_mask` indicates where padding has been applied so that the models ignore it.

**N.B.** This isn't the attention mask as `DistilBERT` is an encoder that doesn't have a `MaskedSelfAttention` layer.

****

### **<u>Model</u>**

When downloading a pre-trained model, we use the `AutoModel` module in the `transformers` library.

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

`AutoModel` module is generic as in, It returns the model without the ***Head***.

![](./Images/Chapter2/transformer_and_head.png)

So, this `AutoModel` models don't specialize in anything.

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape) #torch.Size([2, 16, 768])
```

This returns the output of the hidden states from the picture which we can use with other layer/s to complete our model.

If we want to import the whole thing, we can do

```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape) # torch.Size([2, 2])
```

`AutoModelForSequenceClassification` imports a SequenceClassification specific architecture that only needs to be fine tuned and add `Softmax` for turning logits intro predictions.(**Post processing**)

There are other Application specific modules

- `*Model` (retrieve the hidden states)
- `*ForCausalLM`
- `*ForMaskedLM`
- `*ForMultipleChoice`
- `*ForQuestionAnswering`
- `*ForSequenceClassification`
- `*ForTokenClassification`

****

#### **<u>Accessing model layers</u>**

- :hugs: transformers models behave like **named tuples** or **dictionaries**, as they can be accessed by:
  - **Attributes**: `outputs['last_hidden_state']` 
  - `outputs.last_hidden_state`
  - **Index** `outputs[0]`

****

### **<u>Post-processing</u>**

- We saw that the models outputs logits. which means that they need `softmax` to be turned to probabilities/predictions.

- If you want to know what each `label` stands for use 

  ```python
  model.config.id2label
  # {0: 'NEGATIVE', 1: 'POSITIVE'}
  ```

****

## **<u>Model</u>**

![](./Images/Chapter2/config_model.png)

### **<u>Building a BERT model</u>**

The following will output a model that is not pretrained.

```python
from transformers import BertConfig, BertModel

# Buidling the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
```

```python
print(config)
```

```python
BertConfig {
  [...]
  "hidden_size": 768,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  [...]
}
```

The model above is not pretrained.

To load pretrained model, we use:

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

However this method isn't preferred as changing the model to `DistilBERT` for example would require a lot of code change.

****

#### **<u>Instantiating a model</u>**

```python
from transformers import AutoModel

checkpoint = "bert-base-cased"
model = AutoModel.from_pretrained(checkpoint)
```

This is better as you only change the check point.



