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

- "Jordan scored 30 points", "Jordan has a forecasted temperature of 25 C"
  - I just liked this example and wanted to add it.

### Data Cleaning

- We can use KBs as a form of data cleaning, if you have an erroneus corpus, recognizing the entities and setting constraints can limit the amount of bad data from the erroneus corpus
- e.g. "Jeff Bezos won the Grammy in 2010"
  - This isn't possible as a Grammy is only given to musicians.
- We can then set constraints on specific things that we care about.

#### Constraint Types

- **Type Constraint**: A Grammy winner who belongs to the type `person` must also be of an instance `musician` (or a sub-type)

- **Function dependencies**: There is only 1 Grammy award per year
- **Inclusion dependencies**: All Grammy winners are a subset of people who have atleast 1 song
- **Disjointness Constraint**: (don't really understand it go back to it later.)
- **Temporal Constraint**: The Grammy award can only be given to living people, so a person who won the grammy must be alive.

### Visual Understanding

****

## Scope and Tasks

### Knowledge Base definition

- Throughout the survey, we use the following pragmatic definition of a KB

A Knowledge Base (KB) is a collection of structured data about **entities and relations** with the following characteristics

- **content**: re-read
- **Quality**
  - Have *near-human* quality (i.e. rate of error as if an expert curator was constructing the KB)
  - Should be continuously updated for freshness and maintaned in a consistent way.

- **Schema and Scale**: Constructed with the "pay-as-you-go" principle, which allows ad-hoc additions.
- **Open Coverage**
  - An **ideal** KB would contain all entities and their properties **that are of interest for the domain or enterprise**
  - But practically this isn't possible as most domains are dynamic and always change
  - This results in **KB construction and maintenance** being viewed as a "never-ending" task

****

### Creating a KB

- Steps involved are

#### Discovery

- **Source Selection**
  -  KB Construction starts with deciding on the inteded scope of entities and types that should be covered
- **Content Discovery**
  - Selecting promising sourcs for the entities in the previous step

- **Alignment Computing**
  - ??
- **Entity Detection**
  - Spotting entity mentions in tables, lists or text.
  - Types are discovered from input sources and organized into  a **hierarchical taxonomy**
  - This taxonomy is used to populate and augment the KB at later stages


****

- **Canonicalization**
  - Until this phase, all entities detected are in the form of **mentions**, separate names for the same entity are all unresolved
  - Identifying synonyms (duplicates) and disambiguating mention strings is a **key step** for a high quality KB.
  - When the input is **semi-structured web content** or **unstructured text** and there is already an initial reference of repository of entities, this is called **entity linking**

- **Augmentation**
  - We now have a core KB with **canonicalized entities** and a **clean type system**,
  - The next task is to populate properties with instace-level statements (attribute value and entity pairs that stand in a certain relation) (triplet creation)
  - We do this using the **learning-based extraction properties** of different algorithm
- **Cleaning**
- 

