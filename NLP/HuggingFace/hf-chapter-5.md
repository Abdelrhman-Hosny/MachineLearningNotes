[toc]

# **<u>HuggingFace Datasets</u>**

## **<u>Loading Data not on the Hub</u>**

| Data Format        | Loading script | Example                                         |
| ------------------ | -------------- | ----------------------------------------------- |
| csv & tsv          | csv            | `load_dataset("csv", data_files="file.csv")`    |
| text file          | text           | `load_dataset("text", data_files="file.txt")`   |
| JSON & JSON lines  | json           | `load_dataset("json", data_files="file.json")`  |
| Pickled DataFrames | pandas         | `load_dataset("pandas", data_files="file.pkl")` |

- Each of these has different parameters, so it's advised to look the the documentation for `load_dataset` and check examples before actually trying.

****

## **<u>Data Wrangling</u>**

- In this section, we will show the following operations
  1. Shuffling and Splitting
  2. Selection and Filtering
  3. Rename, remove and flatten
  4. Mapping

****

- Lets first load the dataset from the csv

  ```python
  from datasets import load_dataset
  
  data_files = {"train": "drugComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
  
  drug_dataset = load_dataset("csv", data_files=data_files, delimiter='\t')
  ```

****

- I've found that summarizing the data wrangling part isn't as useful, neither is reading it from the course page/video. The best way to learn it is to actually go with the notebook and then try running it on your own after finishing the notebook
- I'll just write the name of the functions used, as they are pretty self explanatory (more like a cheat-sheet than an explanation).

****

- `drug_dataset.shuffle()`

- `drug_dataset.select(list_of_indices)`

- ```python
  drug_dataset.rename_column(
      original_column_name="Unnamed: 0",
      new_column_name="patient_id"
  )
  ```

- Combining filter with lambda functions

  ```python
  drug_dataset.filter(lambda x: x['condition'] is not None)
  ```

- Combining filter with lambda functions

  ```python
  drug_dataset.map(
      lambda x: {"review_length": len(x['review'].split())}
  )
  ```

- You can add a column using `drug_dataset.add_column()` when `Dataset.map()` doesn't suit your needs.

- `drug_dataset['train'].sort("review_length")`

****

## **<u>Dataset to DataFrame and back</u>**

- Sometimes, more processing is needed than the one available in huggingface. 
- we can use `Dataset.set_format('pandas')`
  - Now, we you access the data inside, a `pandas.DataFrame` will be returned instead of a dictionary
- You can apply whatever analysis/cleaning you want, but before returning to huggingface for training, you have to do `Dataset.reset_format()`
  - If this is not done, you'll run into problems when using HuggingFace models/tokenizers

****

- You can also do your analysis in  pandas from the start and once you get to the model training phase, do the following

  ```python
  from datasets import Dataset
  
  freq_data = Dataset.from_pandas(frequencies)
  ```

****

## **<u>Creating a validation dataset</u>**

- we use `Dataset.train_test_split()` which is in no way related to `sklearn.model_selection.train_test_split()`.

  ```python
  drug_dataset #dataset after tokenizing, feature engineering ..etc
  
  drug_dataset_clean = drug_dataset['train'].train_test_split(train_size=0.8)
  
  drug_dataset_clean['validation'] = drug_dataset_clean.pop('test')
  
  drug_dataset_clean['test'] = drug_dataset['test']
  ```

****

## **<u>Saving a dataset</u>**

| Data format |         Function         |
| :---------: | :----------------------: |
|    Arrow    | `Dataset.save_to_disk()` |
|     CSV     |    `Dataset.to_csv()`    |
|    JSON     |   `Dataset.to_json()`    |
|   Parquet   |  `Dataset.to_parquet()`  |

****

### **<u>When to use each ?</u>**

- CSV and JSON are used when you don't have a lot of data.
- Arrow and Parquet are used when data is bigger.
- Arrow is used when you'll reuse the data again soon.
- Parquet is used when the data will be stored for a while and it is very space efficient.

****

## **<u>Loading a dataset</u>**

- Used with arrow files

  ```python
  from datasets import load_from_disk
  
  drug_dataset_reloaded = load_from_disk("drug-reviews")
  ```

- for csv and json, we use the ones mentioned at the start of the document.

  ```python
  load_dataset('csv',...)
  load_dataset('json',...)
  ```

****

## **<u>Dealing with "Big data"</u>**

### **<u>Memory Mapping</u>**

- `Dataset` uses **virtual memory** in order map each portion of the dataset to a certain part of memory, and only load the needed parts to RAM when necessary.
- This allows us to deal with datasets that are larger than the size of our RAM.
- To properly understand this, you have to understand what is **virtual memory**.

****

### **<u>Streaming</u>**

- If the dataset is very large even for **memory mapping**, you get the option to **stream the data**.

- This puts a pointer at the first location of the data in memory and you can increment that pointer as needed using `IterableDataset.take()` and `IterableDataset.skip()`

  - **N.B.**   `IterableDataset` is the class used for streamed datasets.

- You can stream data by

  ```python
  dataset_streamed = load_dataset(...., streaming=True)
  ```

  Now `dataset_streamed` is of type `IterableDataset`, and you can access the first element using

  ```python
  next(iter(dataset_streamed))
  ```

- We can still use `.map()` in the same way.

  ```python
  from transformers import AutoTokenizer
  
  tokenizer = AutoTokenizer.from_pretrained(...)
  tokenized_dataset = dataset_streamed.map(lambda x: tokenizer(x['text']), batched=True)
  next(iter(tokenized_dataset))
  ```

- We can also use `.shuffle()` but it only shuffles in a predefined `buffer_size`

  ```python
  shuffled_dataset = dataset_streamed.shuffle(buffer_size=1000)
  ```

  This only shuffles the first 1k elements and leaves the rest as is.

****

#### **<u>Combining Streamed Datasets</u>**

- This can be done using `interleave_datasets` from `datasets` library

  ```python
  from itertools import islice
  from datasets import interleave_datasets
  
  combined_dataset = interleave_datasets([dataset1_streamed, dataset2_streamed])
  list(islice(combined_dataset, 2))
  ```

  ```python
  [
      {'meta': {'pmid': 11409574, 'language': 'eng'},
    	'text': 'Epidemiology of hypoxaemia in children with 				acute lower respiratory infection ...'},
   
   {'meta': {'case_ID': '110921.json',
     'case_jurisdiction': 'scotus.tar.gz',
     'date_created': '2010-04-28T17:12:49Z'},
    'text': '\n461 U.S. 238 (1983)\nOLIM ET AL.\nv.\nWAKINEKONA\nNo. 81-1581......'}
  ]
  ```

  We've also used the `islice(iterable, start, stop, step)` from `itertools` module.

  This function take an iterable and returns certain elements from it based on the three arguments `start, stop, step`.

****

## **<u>Creating a dataset</u>**

- This won't be of use to me at the moment, so I'm skipping it sorry :(

****

## **<u>Semantic Search with FAISS</u>**

- This will involve some of the stuff used in **<u>Slicing and Dicing</u>**, I'm only going to document new stuff.

****

### **<u>Using Embedding for semantic search</u>**

- Transformer--based language models represent each token in a span of text as an **embedding vector**. (there is a vector for each word, the size of that vector depends on the model being used)
  - We can "pool" that embedding vector to obtain a 1- representation vector for **each sentence**
  - We can later use that vector to find the similarity between sentences using methods like dot product similarity.

****

- Keeping only specific columns in a dataset

  ```python
  columns = dataset.column_names
  cols_to_keep = ['title', 'body', 'html_url', 'comments']
  cols_to_remove = set(cols_to_keep).symmetric_difference(columns)
  dataset = dataset.remove_columns(cols_to_remove)
  dataset
  ```

  ```python
  Dataset({
      features: ['html_url', 'title', 'comments', 'body'],
      num_rows: 771
  })
  ```

- In the dataset above, each `body` has several `comments`; a list of comments to be precise.

  We want to turn each comment into a row.

  ```python
  dataset.set_format('pandas')
  df = dataset[:]
  comments_df = df.explode("comments", ignore_index=True)
  ```

****

### **<u>Creating Text embeddings</u>**

- We are first going to download the model called `multi-qa-mpnet-base-dot-v1` as if you check the docs, this one performs the best in the case of semantic search.

  ```python
  from transformers import AutoTokenizer, AutoModel
  
  model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
  tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
  model = AutoModel.from_pretrained(model_ckpt)
  ```

  To speed the process, we'll add the model to gpu

  ```python
  import torch
  device = torch.device('cuda')
  model.to(device)
  ```

- We've mentioned before that we'd like to do **pooling** in order to embed the whole sentence into a vector

  ```python
  def cls_pooling(model_output):
  	return model_output.last_hidden_state[:, 0]
  ```

- Next, we wrap the whole process in a function `get_embeddings(text_list)`

  ```python
  def get_embeddings(text_list):
  	encoded_input = tokenizer(
      	text_list, padding=True, truncation=True, return_tensors="pt"
      )
      encoded_input = {k: v.to(device) for k,v in encoded_input.items()}
      model_output = model(**encoded_input)
      return cls_pooling(model_output)
  ```

- We can apply this to our dataset using the `map` function

  ```python
  embeddings_dataset = dataset.map(
  	lambda x: {"embeddings": get_embeddings(x['text']).detach().cpu().numpy()[0]})
  ```

****

### **<u>Using FAISS for efficient similarity search</u>**

- FAISS stands for Facebook AI Similarity Search.

  - FAISS is a library that provides **efficient algorithms to quickly search and cluster embedding vectors**.

  - `Dataset` class contains what is called a FAISS index

    - we can add it using `Dataset.add_faiss_index()`

      ```python
      embeddings_dataset.add_faiss_index(column="embeddings")
      ```

- We can now perform queries on this index by doing a **nearest neighbor lookup** with `Dataset.get_nearest_examples()`

  ```python
  question = "How can I load a dataset offline?"
  question_embedding = get_embeddings([question]).cpu().detach().numpy()
  question_embedding.shape # [1, 768]
  ```

  Now that we have the embedding

  ```python
  scores, samples = embeddings_dataset.get_nearest_examples("embeddings", question_embedding, k=5)
  ```

- We now have the closest 5 possible answers stored inside `samples` which is a dictionary. and the measure of closeness inside the `scores`

  - We are going to turn it into a pandas dataframe and get the closest answer.

    ```python
    import pandas as pd
    
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df['scores'] = scores
    samples_df.sort_values('scores', ascending=False, inplace=True)
    ```

    Now we can iterate over them

    ```python
    for _, row in samples_df.iterrows():
        print(f"COMMENT: {row.comments}")
        print(f"SCORE: {row.scores}")
        print(f"TITLE: {row.title}")
        print(f"URL: {row.html_url}")
        print("=" * 50)
        print()
    ```

****

**N.B.** For **asymmetric** semantic search, we usually have a **short query** and a **longer paragraph**.

****

