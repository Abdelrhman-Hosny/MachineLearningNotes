# 8 - Basics
Cluster = Collection of (related) nodes
Node = Instance of elastic search (multiple nodes allow for quicker search)

- When a new node is created, it always joins a cluster
  - It creates a new cluster unless configured to join an existing one

## Storing data

Data is stored as documents (JSON objects)

When storing data in elastic

```
data = {
    "name": "Hosny",
    "county": "Egypt",
}
```

Elastic stores some meta deta along with the document

```
stored_document = {
    "_index": "people",
    "_type": "_doc",
    "_id": "123",
    "_version": 1,
    "_seq_no": 1,
    "_primary_term": 1,
    "_source": {
        "name": "Hosny",
        "county": "Egypt",       
    }
}
```

### Index

Index is where you store documents

# 9 - Inspecting cluster

dealing with cluster using `_cluster` api

- There is also an API called `_cat` which outputs data in human readable format

# 11 - Sharding

- Sharding is a way to divide indices into smaller pieces
- Each piece is referred to as a shard
- sharding is done at the **index level**

Main reason for sharding is **horizontal scaling**

- A shard is an **independent <u>Apache Lucene Index</u>**
  - So an Elasticsearch index **consists of** one or more Lucene indices
- A shard has no predefined size, and grows as we add more documents to it.
- A shard may store up to **two BILLION documents**

- To increase/decrease the number of shards, we use the Split/Shrink APIs respectively.
- The default number of shards is 1.
  - If you know that your index will grow, you should add multiple shards from the beginning.

# 12 - Replication


# Queries

- get all indices
  `GET /_cat/indices?v` 

- get data from index
  `GET /index_name/_search` 