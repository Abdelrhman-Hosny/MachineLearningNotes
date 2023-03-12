## meta-data after POST

```
{
  "_index": "products",
  "_id": "Wc0VxoYBClvYlxxL47iF",
  "_version": 1,
  "result": "created",
  "_shards": {
    "total": 3,
    "successful": 1,
    "failed": 0
  },
  "_seq_no": 0,
  "_primary_term": 1
}
```
If we look at shards

```
{
  "_shards": {
    "total": 3,
    "successful": 1,
    "failed": 0
}, 
```

# 4 - Updating the document

## The Update API

- Documents in elastic are immutable
  - When updating, the document is retrieved, then perform the update
  - After that you replace the old document with the updated one.

# 5 - Scripted Updates

```
POST /products/_update/100
{
    "script": {
        "source": "ctx._source.in_stock--"
    }
}

```
# 6 - Upserts


# Watch and write the rest lma tb2a ader

# Queries

- Deleting an index
  `DELETE /index_name`
- creating an index
  ```
  PUT /index_name
  {
    "settings":
        "number_of_shards": 2,
        "number_of_replicas": 2
  }
  ```  
  if we only use `PUT /index_name`, we would use the default settings

- Indexing document
    ```
    POST /index_name/_doc
    {
        "name": "Coffee Maker",
        "price": 64,
        "in_stock": 10
    }
    ```

    If we add a document to a **non-existing index**, the **default** behavior is that it is **automatically created**

- Get
  `GET /index_name/doc_id`
  If document is not found there will be no `_source` field in the returned json.

- Updating
  - An update could mean chaning an **existing** value, or **adding a new one**.
  ```
    POST /index_name/_update/doc_idx
    {
        "doc": {
            "field_to_change_1": new_value_1,
            "field_to_change_2": new_value_2,
            ..
        }
    }
  ``` 
