# Knowledge Graphs (Knowledge Bases)

## Examples on data in KBs

- If you google eminem songs, you'll have KBs containing (at least) the following backbone info
  - Entities like **people, places, orgs, products and events**
    
    e.g. Eminem, Detroit, Abu Dhabi Tour, 2015 Grammy
  - The semantic classes to which entities belong, such as
    
    e.g. <Eminem, type, songwriter>, <Eminem, type, Rapper>

  - Relationships between entities.
    
    e.g. <Eminem, created, My Name Is>
  - Some KBs also contain validity times such as
    
    e.g. <Eminem, married to, Kim, [1988, 2000]>
  
    This **temporal scoping** is optional, but important for the life-cycle management of a KB

****

## Application use cases

### Semantic Search and Question Answering

- Whenever a search engine detects that a user's information need centers around an entity 
  or a specific type of entities, the KB can return a precise and concise list of entities, such as
  singers, songs...etc.

- KBs allow the search engine to return related links instead of outputing "ten blue links".

- Even if the query is too complex for the KB, or if the KB doesn't have enough information, the KB info
  can help to improve the ranking of web-page results by considering the **types** and other properties of entities.

### Language Understanding and Text Analytics

- We can identify mentions of products (and associated customer opinions), link them to a KB
  and then perform comparative and aggregated studies.

- We can even incorporate filters and groupings on product categories, geographics regions by combining
  the textual info with the structured data from the KB.

- A trending example of semantic text analytics is detecting gender bias in news and online content.
  
  Use the KB to compute statistics for male vs female people.


### Data Cleaning

### Visual Understanding

****

## Scope and Tasks

### Knowledge Base definition

- Throughout the survey, we use the following pragmatic definition of a KB

A Knowledge Base (KB) is a collection of structured data about **entities and relations** with the following characteristics

- **content**


