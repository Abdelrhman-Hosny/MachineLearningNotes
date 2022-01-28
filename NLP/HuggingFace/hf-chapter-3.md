[toc]

# <u>**Chapter 3**</u>

## **<u>HuggingFace Datasets</u>**

- HuggingFace Datasets are dictionaries with added functionalities.
- They also have the advantage **being stored on disk** not RAM.
  - Only the examples needed are added to RAM.

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

After loading the dataset, you can see that the `DatasetDict` has 3 `Dataset` instances in it.

`train`, `validation` and `test`.

It also shows the attributes (columns) inside the dataset.

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

From the output above, we can conclude that each instance in the dataset has 4 attributes. 

[`sentence1`, `sentence2`, `label` , `index`] which can be used as needed.

****

#### **<u>Accessing Rows and Columns</u>**

- The `Dataset` Class allows you to access rows and columns. 
  - `raw_dataset[0]` returns the first row (including all 4 attributes).
  - `raw_dataset["sentence1"]` returns the attribute `sentence1` for **all rows**.
- You can mix and match between accessing rows and columns
  - `raw_dataset[1]["sentence1"]` returns `sentence1` of the second row,
  - `raw_dataset["sentence1"][1]` returns the first row of all the `sentence1` instances in the dataset.
  - `raw_dataset[1]["sentence1"] == raw_dataset["sentence1"][1]` 
    - i.e. both expressions are equal.
- You can treat `Dataset` as a normal python list **when dealing with <u>rows</u>** when it comes to indexing
  - `raw_dataset[:5]` returns the first 5 rows.

****

#### **<u>`Dataset.features`</u>**

using `Dataset.features` shows more details about the dataset attributes.

```python
raw_train_dataset = raw_datasets['train']
raw_train_dataset.features
```

You can see the data type of each attribute.

If you check the `label` attribute, you can deduce that the models that use this dataset use it to check whether `sentence1` and `sentence2` are equivalent.

```python
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
```

****

## <u>**Processing a dataset**</u>

The Processing part that is related to HuggingFace is  done mainly by the **tokenizer**.

We've already seen how to process simpler inputs like text classification using **1 sentence**. However as in this dataset there are two sentences that are classified, we will change our pipeline just a small bit.

### **<u>Processing sentence pairs</u>**

- If you look at the [documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) of the `__call__` function of the `tokenizer` and check the first two parameters of the function.
- You'll find that the first one is `text` and the second is `text_pair`. so the tokenizer is already equipped to handle sentence pairs.
- We **will** discuss the **output** when there are two sentences and some **tricks** to use when dealing with sentence pairs.
- We **won't** discuss the tokenizer in detail now as it is covered in a later chapter.

```python
from transformers import AutoTokenizer

checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inputs = tokenizer("This is the first sentence.", "This is the second one.")

print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
print(inputs["token_type_ids"])
```

```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```

- Before seeing the output as a whole, we will first understand what `token_type_ids` represents.
- The way the models handles the sentence pair is that it concatenates them into a single sentence along with adding **special tokens**. 
  - `[CLS] sentence1 [SEP] sentence2 [SEP]`
- How does the model know which sentence is the first and which is the second?
  - It does so using `token_type_ids`, if you look at the output above.
  - You can see that `token_type_ids` have value of `0`, when the word/token belongs to the first sentence (including `[CLS]` and the first `[SEP]` token)
  - When the second sentence starts `token_type_ids` take the value of `1`.
- To summarize, if the word/token belongs to the first sentence, corresponding `token_type_ids` will have a value of `0` and if word/token belongs to the second sentence, the value will be `1`.

```python
print(inputs)
```

```python
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

When we print the `inputs` (output of the tokenizer). we will see that we know everything inside it.

**N.B.** The results may be different depending on the **model** used as each model has it's own way of doing things. but as long as the same checkpoint is used for tokenizer and model, it'll be fine.

****

Now we know how to process **1 example** from the dataset, but how do we apply it to the **dataset as a whole** ?

- One way to do this is to do the following

```python
tokenized_dataset = tokenizer(
	raw_datasets["train"]["sentence1"],
	raw_datasets["train"]["sentence2"],
	padding=True,
	truncation=True,
	)
```

- This has some downsides
  1. Only works if you have enough RAM for the whole dataset
     - As output is now a `Dictionary` instead of `Dataset`
  2. Loses the organization and ease of use of the `Dataset` class as It is turned into a `Dictionary`

****

- To keep the object a `Dataset`  class, we use the `Dataset.map()`  function.
- `Dataset.map()`  applies a function on each row of the dataset.

```python
def tokenize_function(example):
	return tokenizer(example['sentence1'], examples['sentence2'], truncation=True)
```

And then use

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

This `map` function takes the output that is returned and **adds** to the `features` of the dataset

```python
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```

**N.B.** `raw_datasets.map(.... , batched=True, num_proc=2)`

- `batched=True` allows use to run the map function on multiple elements at a time instead of one by one
- `num_proc=2` allows use to run 2 processes to `map` the data faster.

****

## <u>**Static vs Dynamic Padding**</u>

#### **<u>TL;DR</u>**

- Use Dynamic Padding when using CPU or GPU
- Use Static Padding when using TPU

****

### **<u>Static Padding</u>**

Useful for TPU Training. 

#### **<u>TPU Training</u>**

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

### **<u>Dynamic Padding</u>**

Useful for GPU & CPU Training

#### **<u>GPU & CPU Training</u>**

Dynamic is when you pad all the sentence up to the longest sentence in **each batch**.

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

#### **<u>Data Collators</u>**

Data collators are objects that will **form a batch** using a list of dataset elements as input.

Data Collators may apply some **preprocessing** like **padding** or **random data augmentation**.

****

Padding in HuggingFace can be done using the `DataCollatorWithPadding` function that takes the tokenizer as input.

- The data collator knows which type of padding to apply from the `tokenizer`.
  - `tokenizer(example["sentence1"], example["sentence2"],truncation=True, max_length=70, padding="max_length")` will apply **static padding** with size 70.
  - `tokenizer(example["sentence1"], example["sentence2"], truncation=True)` will apply dynamic padding.

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

We can also apply the `data_collator` directly on the data.

```python
samples = tokenized_datasets['train'][:8]
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

```python
{'attention_mask': torch.Size([8, 67]),
 'input_ids': torch.Size([8, 67]),
 'token_type_ids': torch.Size([8, 67]),
 'labels': torch.Size([8])}
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

####  Evaluation Strategy

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

