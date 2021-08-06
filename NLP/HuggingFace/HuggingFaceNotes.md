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

You can modify the config to fit your needs (i.e. set `num_hidden_layers = 10`)

```python
from transformers import BertConfig, BertModel

# Buidling the config
config = BertConfig.from_pretrained('bert-base-uncased', num_hidden_layers=10)

# Building the model from the config
model = BertModel(config)
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

****

#### <u>**Save & Load models**</u>

```python
model.save_pretrained('directory_on_computer')
```

```bash
ls directory_on_computer
# config.json  pytorch_model.bin
```

The config.json knows the architecture of the saved model while the bin file has the weights

****

## **<u>Tokenizers</u>**

### <u>**Word-based tokenization**</u>

Discussed in CS224n

****

### **<u>Character-based tokenization</u>**

- **<u>Pros</u>**
  1. Much smaller vocabulary
  2. Less oov words since every word can be built from characters
- **<u>Cons</u>**
  1. Lose on semantic meaning of the words
  2. Very long input which may increase likelihood of vanishing/exploding gradients
  3. Length of a sentence will be very large.

****

### <u>**Subword-based tokenization**</u>

This gets the best of both words from Character and Word based tokenizers.

- This makes it so that common words are not split while **rare words** are decomposed.

  i.e. annoyingly -> annoying & ly where each of these have a meaning

This is especially useful in languages where you can form words by putting multiple words together (Arabic & German).

****

### **<u>Loading & saving tokenizers</u>**

#### Loading

```python
from transformers import BertTokenizer, AutoTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
```

```python
tokenizer('Using a Transformer network is simple')
```

```python
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

#### Saving

```python
tokenizer.save_pretrained('directory_on_pc')
```

****

Steps done by tokenizer are broken into several functions in case you want to use them separately.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

```python
# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
```

**N.B.** BERT uses ## to indicate that this is not a start of a word.

```python
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# [7993, 170, 11303, 1200, 2443, 1110, 3014]
```

```python
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
# 'Using a Transformer network is simple'
```

****

## <u>**Transformers for Long Sequences**</u>

Models have different supported sequence lengths, and some specialize in handling very long sequences. [**Longformer**](https://huggingface.co/transformers/model_doc/longformer.html) is one example, and another is [**LED**](https://huggingface.co/transformers/model_doc/led.html). If you’re working on a task that requires very long sequences, we recommend you take a look at those models.

****

# <u>**Chapter 3**</u>

## **:hugs: <u>Datasets</u>**

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

```python
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

```python
raw_train_dataset = raw_datasets['train']
raw_train_dataset.features
```

```python
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
```

****

## <u>**Processing a dataset**</u>

### Processing sentence pairs

```python
from transformers import AutoTokenizer

checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```

```python
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

`token_type_ids` tells the model which tokens are part of the first sentence and which are part of the second one.

```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```

****

### Dataset.map

This applies a function to each element in the dataset and is useful in preprocessing.

****

## <u>**Static vs Dynamic Padding**</u>

Padding in :hugs: can be done using the `DataCollatorWithPadding` function that takes the tokenizer as input and knows from it what type of padding to apply.

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

and then this is passed to `DataLoader`

```python
train_dataloader = DataLoader(
	tokenized_datasets['train'], batch_size=16, collate_fn=data_collator
)
```

### Static

Useful for :arrow_down_small:

#### TPU Training

Static is when you pad all the sentences to the same length for **all batches**

```python
for i, batch in enumerate(batches,1):
	print(f'Batch {i} shape : ', batch.shape)
# (num_sentences, sentence_length)
# Batch 1 shape : (4, 16)
# Batch 2 shape : (4, 16)
# Batch 3 shape : (4, 16)
# Batch 4 shape : (4, 16)
```

So each one of our sentences will be padded or truncated to 16 no matter how long it is.

****

### Dynamic

Useful for :arrow_down_small:

#### GPU & CPU Training

Dynamic is when you pad all the sentence up to the longest sentence in each batch.

```python
for i, batch in enumerate(batches,1):
	print(f'Batch {i} shape : ', batch.shape)
# (num_sentences, sentence_length)
# Batch 1 shape : (4, 20) # max sentence len = 20
# Batch 2 shape : (4, 10) # max sentence len = 10
# Batch 3 shape : (4, 8)  # max sentence len = 8
# Batch 4 shape : (4, 11) # max sentence len = 11
```

****

## <u>The `Trainer` API</u>

![](./Images/Chapter3/trainer-api.png)

```python
from transformers import AutoModelForSequenceClassification
# old stuff
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
			'test-trainer',
			per_device_train_batch_size=16,
			per_device_eval_batch_size=16,
			num_train_epochs=5,
			learning_rate=2e-05,
			weight_decay=0.01,
)
```

```python
from transformers import Trainer

trainer = Trainer(
			model,
			training_args,
			train_dataset=tokenized_datasets['trained'],
			eval_dataset=tokenized_datasets['validation'],
			data_collator=data_collator,
			tokenizer=tokenizer
)

trainer.train()
```

#### Evaluation Strategy

```python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch") # tells model to evaluate each epoch
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics # tells model which metrics 
)									# to use during evaluation
```

****