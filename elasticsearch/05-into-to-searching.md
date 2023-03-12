
- Elasticsearch uses BM25 when computing the score for text fields

# Understanding Relevance Score

- Debugging for a whole query


```json
GET /index_name/_search?explain=true
{
    "query": {
        "term": {
            "name": "lobster"
        }
    }
}
```

# 8 - The `_explain` API

- Debugging 1 example

```json
GET /<index>/_explain/<doc_id>
{
    "query": {
        "term": {
            "name": "lobster"
        }
    }
}
```

# 9 - Query vs Filter Context

- Major difference is that Query **calculates relevance score**


- Use `"query"` clause when you want relevance score
- If you want to include all docs that include a word use the `"filter"` clause.

****

# 10 - `term` vs `match` queries

- When you use `match`

    - ```json
      GET /product/_search
      {
        "query": {
            "match": {
                "name": "Lobster"
            }
        }
      }
      ```
      The field `"name"` is passed through **the** **same** **analyzer** used to analyze the field before indexing

- When you use `term`
    - ```json
      GET /product/_search
      {
        "query": {
            "term": {
                "name": "Lobster"
            }
        }
      }
      ```
      The field `"name"` is passed through **no-op analyzer** used to analyze the field before indexing


- Analyzer could be considered as text preprocessing (go to ch4 to learn more)

****