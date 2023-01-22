# KB Construction: Attributes and Relationships

## Problem and Design Space

- From the last two chapters, we've created and canonicalized entities and classes.

- The next step is to enrich the entities with **properties** in the form of triplets, covering **both**:
  - **Attributes**: Relationship between **entity** and a **literal** value
    - e.g. year when album was released, maximum speed ..etc
  - **Relations**: Relationship between **two entities**
    - e.g. "Eminem" is a "rapper", "Eminem" is a "songwriter" ..etc

****

- Methods for extracting SPO triplets can handle both att.s and relations in a more or less unified way

****

**Assumptions**

- Best-practice methods are built on a number of **justified assumptions**

**Argument Spotting**

- Given a text, we can **spot and canonicalize** arguments for the **subject and object** of a candidate triple
- This assumption is valid as we already have methods for **entity discovery and linking**
- We also have specific techniques for dates and number that can be harnessed for spotting and normalization


**Target Properties**

- We assume that we know the type of properties that we want to capture.
  - e.g. for people, we are intersted in birthdate, birthplace, name ...etc
  - for musicians, we need songs composed, albums released ...etc

**Type Signatures**

- We assume that each property has a type.
- This is part of the KB schema (ontology)

****

### Schema Repository of properties

- There are resources that provide properties and type signatures for classes

**Examples**

- Early KB projects like Yago and Freebase
- frameworks like schema.org

****

## Pattern-based and Rule-based Extraction

- The easiest source for doing this are also premium sources, think wikipedia tables.

- For the premium sources we can use simple methods to extract relations

### Regex Patterns

e.g. `birth year X: Born .* (X = (1|2)[0-9]{3}) .*`

- There are also ways to induce Regex Patterns from examples.

****

### Type Checking

- We can use type checking to remove the extracted false positive relations

****

### Operator-based Extraction plans

- This is essentially having a small quick pipeline of several rules to check for a relation.

e.g.

We want to extract all triples of the form `playsInstrument: musician x instrument`
- We also want to check that this is a live performances
- We would have an algorithm that goes like

```
1. Detect all person names in the text
2. Filter out non musicians, by entity linking and type checking (or accept them as out-of-KB entities)
3. Detect all mentions of musical instruments (using same process above) 
4. Check pairs of instruments and musicians that appear in the same sentence (should apply co-reference resolution)
(sometimes deeper analysis is called for) 
5. Check that entire text refers to a live performance.
```

****

### Patterns for Text, Lists and Trees

- e.g. `P * born in * C` can indicate birthplace of the person

- Dependency Parsing and Coreference Resolution are used in this process

****

## Distantly Supervised Pattern Learning

- Rule-based methods often get high precision but are not suitable if you're aiming for high recall


### Statement-Pattern Duality

```
Statement-Pattern Duality

When correct statements about P(S,O) for property P frequntly
occur with textual pattern p, then p is a good pattern for P

Conversely, when snipper with two arguments P and O contain a good pattern
p for property P, then the stepment P(S,O) is likely correct
```

#### Pattern Discovery

- Find occurrences of $(S_i, O_i)$ pairs from $T$ in a web corpus and identify new patterns $p_j$ that co-occur with these pairs with high frequency (and other statistical measures), and add $p_j$ ti $P$.

#### Statement Expansion

.. to be completed once I finish chapter overview if needed.

****

### Quality Measures

This is where you stopped last time.

****

