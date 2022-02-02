[toc]

# ML Design Patterns

# Chapter 1

# Common Challenges in ML

## 1. Data Quality

### <u>Garbage In, Garbage out</u>

****

### <u>Data Completeness</u>

Make sure that the data represents all possible test inputs.

i.e. if you train the model on images of cats and dogs, and the test input is an image of a horse, it'll still be classified as cat or dog.

The "world" of the model doesn't contain horses.

****

### <u>Data Consistency</u>

Since the labelers of the data are usually different people, each of them may label the data differently.

Use Celsius on some examples and Fahrenheit on others.

When adding bounding boxes on cat images, some people may not include the tails in the bounding box.

**Timeliness** in data refers to the **latency** between when an event occurred and when it was added to the database which may cause the model to be trained on old data.

****

## Reproducibility

ML models inherently contains a lot of random things like seeds and weight initialization. so you have to try as much as possible to set the same seeds for all the random operations to make your results reproducible.

This may also include remembering the dependencies of your project and their versions. `tf2.1` may produce different results than `tf2.2`

****

## Data Drift

This refers to training a model on old data or the decaying of model performance over time due to the introduction of new types of data.

****

## Scale

****

## Multiple Objectives

Each team may have different vision for what you call a "successful" model.

ML Engineer will focus on minimizing the cross entropy loss, but a team manager may want the model to decrease the number of misclassified examples by 5x

****

