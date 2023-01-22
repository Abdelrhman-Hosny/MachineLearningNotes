# Knowledge Integration from Premium Sources

- Here we focus on the knowledge harvested from premium sources such as Wikipedia or domain-specific repositories such as:
  - GeoNames for spatial entities
  - GoodReads and Librarything for the domain of books


## The case for premium sources

- Paper authors recommend starting every KB construction project by tapping one or a few premium sources first, such sources should have the following characteristics
  - Authoritative high-quality content about entities
  - high coverage of many entities
  - clean and uniform representation of content, like having clean HTML markup or wiki markup ..etc

****

- Distilling from such sources can create a **strong core KB** with a good yield-to-effort ratio.
  - As with such clean data, you can use relatively simple methods and get great results
- From there you can use more complex methods in order to expand upon the core KB

****

- You should take care that the same entity might exist in different premium sources
  - So simply taking the union of these KBs is not a viable KB
  - We would need to do **entity matching** in order to align these entities.

****

## Category Cleaning

- Premium sources come with a rich **category system** assigning pages to relevant categories that can be vieweed as proto-classes but are **too noisy to be considered a semantic type system**


- If we consider Wikipedia, their category system is **almost** a class taxonomy, However we face the following difficulties:
  - **High Specifity of Categories**
    - The classes are too specific e.g. "American male singer-songwriters", "20th-Century American guitarists"
    - The computer just sees these as noun phrases.
  - **Drifting Super-Categories**
    - By considering also super-categories
  - **Entity Tpes vs Associative Categories**
    
All of the above makes sense when the category hierarchy is viewed as **means to support user browsing**, but it is not acceptable for the **taxonomic backbone of a clean KB**

****

### Category Cleaning Algorithm

```python

def category_cleaning(
    c: CategoryName
) -> Union[str for semantic class label, None]:

   # run noun-phrase parsing to identify head_word and modifier structure

   # Test if head_word is in plural form or has a frequently occurring plural form
        # if not return null
        # Optionally consider pre_i ... pre_k head_word as class candidates, 
        # with increasing i from 0 to k-1

    # for leaf category c, return head_word (optionally pre_1 pre_2 .. head_word)
    # for non-leaf category c, test if the class candidate head_word is a synonym or hypernym (i.e. generalization) of an already accepted class including head_word)

```

****

## Alignment between Knowledge Sources

****

# Take-Home Lessons (Summary)

- For building a core KB, with individual entities organized into a clean taxonomy of semantic types, it is often wise to start with one or a few premium resources.
  - Examples are Wikipedia, GoodReads, IMDB, WikiVoyage ..etc

- A key asset are **categories**, in these sources categories are designed for **manual browsing**, this typically involves a **category cleaning step** to identify **taxonomically clean classes**

- To construct an expressive and clean taxonomy, while havesting two or more premium sources, it is often necessary to **integrate different type systems.**
  - This can be achieved by **alignment heuristics** based on NLP techniques (such as **noun-phrasing**), or by random walks over candidate graphs.

****
