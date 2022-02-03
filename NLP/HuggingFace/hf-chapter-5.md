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

