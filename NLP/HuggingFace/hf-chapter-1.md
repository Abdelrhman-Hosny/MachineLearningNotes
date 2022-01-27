# <u>**HuggingFace Notes**</u>

# <u>**Chapter 1**</u>

## The `pipeline()` function

It is the highest level function in huggingface. It takes care of everything.

Just give the function raw text and it will generate the output for the specified task

### Examples

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvian ,and I work at Hugging Face in Brooklyn")
```

This takes the text and passes it through a certain model along with all the required pre and post processing steps needed for that specific model.

You can change the model as a parameter for the function. 

`pipeline()` takes as input the desired task `"ner"` in our case and can take as input a `model` parameter.

i.e. `translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")`

### Limitations

The pipelines shown are mostly for demonstrative purposes. They can only perform specific tasks and cannot perform variations of them.

****

## <u>Types of Transformers</u>

### **<u>Encoder</u>**

- Only uses the encoder part of the transformer.

- If the input to the encoder was `n` words, the output would be `n` vectors.

  Each vector  $v_i$ is a **numerical representation** of the word $i$, but in the **context** of the surrounding words.

  **Context** in encoder is **Bi-directional** (i.e. context from words before and after).

  **Context** is computed by the **Self-Attention** mechanism

  length of vector $v_i$  depends on the model.

- Used when you need to understand the contents of the sentence like **sequence classification, question answering & masked language modeling**.

- Examples Include: **BERT, RoBERTa, AlBERT & DistilBERT**

****

### **<u>Decoder</u>**

- Only uses the decoder part of the transformer and are also called **auto-regressive** models.

  **Auto-regressive**: re-uses last input as output in the next step.

- Decoders can do the same job as encoders, but with **loss** of performance.

  Just like the encoder, the decoder outputs `n` vectors if given `n` words.

  The difference is that decoders use **Masked attention mechanism** instead of **self attention mechanism**. This causes the **context** to come from the **words before only**.

- Best suited for **text generation** as they revolve around predicting the **next word in a sentence**. Use in tasks where you would only look at the left of the word when predicting.

- Examples include: **GPT, GPT-2, CTRL & Transformer XL**

****

### **<u>Seq2Seq</u>**

- Use both the encoder and decoder parts of the transformer.
- Suitable for tasks revolving around **generating new sentences** depending on a **given input** (**conditional text generation**)
- Tasks include: **Machine Translation**, **Summarization** and **generative question answering**
- Examples include: **BART, mBART, Marian & T5**.

****

Take care as transformers learn the biases of the data it was trained on.

> 