# KB Construction: Entity Discovering and Typing

- This chapter presents advanced methods for populating a knowledge base with entities and classes (aka types), by tapping into **textual and semi-structured sources**

- This chapter will discuss how to extract entites, and the next chapter will discuss how to canonicalize them.

****


Premium sources alone are typically insufficient to capture
  - long-tail entities, such as less prominent musicians, songs and concerts
  - long-tail classes, such as left-handed cello players

This chapter presents a suite of methods for automatically extracting such additional entities and classes from sources like webpages and text documents.

****

- Since we already have a **core KB** from premium sources, we leverage that and turn this into several discovery tasks:
    1. Given a class $T$ containing a set of entities $E = \{e_1 ... e_n\}$, find more entities for $T$ that are not yet in $E$.
    2. Given an entity $e$ and its associated classes $T = \{t_1...t_k\}$, find more classes for $e$ that are not yet captured in $T$
    3. Given an entity $e$ with known names, find additional (alternative) names for $e$, such as acronyms or nick names
        - This is often referred to as **alias name discovery** 
    4. Given a class $t$ with known names, find additional names for $t$.
       - This is sometimes referred to as **paraphrase discovery** 

Methods used on one task can apply to another, so paper is not organized by task.

There is also a case where we **don't have a premium source** to be harvested first. This case is called **ab-initio taxonomy induction** (or **zero-shot learning**)

****

## Dictionary-based Entity Spotting

