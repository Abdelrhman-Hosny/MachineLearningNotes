[toc]

# ML Design Patterns

# Chapter-2 : Data Representation

In this book, *input* is used to refer to raw input data, while *feature* is used to refer to **transformed data**.

**embeddings** is an example of the *automatically-learnable* data representations (automatically engineered features).

Data representations doesn't need to be of a **single input variable**, Models like *oblique decision trees* that create a node based on **a linear combination of multiple variables**.

****

# Simple Data Representations

## <u>Numerical Inputs</u>

One of the steps of using numerical inputs is **scaling**.

- **Scaling** makes the optimization quicker as it helps the loss functions to be more spherical and of nearly equal magnitude in all directions, which **decreases the steps taken to reach the global minima**.
- **Scaling** also helps with regularization, as they depend  on the magnitude of the weights which would cause **different features to be affected differently by regularization**

****

## **<u>Linear Scaling</u>**

### <u>**Min-max scaling**</u>

The numeric values is scaled so that $x \ \epsilon \ [-1,1]$.
$$
x_\text{scaled} = \frac{2x - max(x) - min(x)}{max(x)-min(x)}
$$
the problem with min max scale is that:

- $max(x)$ and $min(x)$ from the train set are often outliers.
- so the rest of the data gets shrunk to a very narrow range in the $[-1,1]$ band.

****

### <u>Clipping + Min-max scaling</u>

The numeric value is scaled between two **reasonable bounds**.

non outliers lie in the range of $] -1,1[$ and all outliers are either $-1$ or $1$.

The negative side of this is that all outliers are treated the same.

****

### <u>**Z-score normalization**</u>

$$
x_\text{scaled}= \frac{x - E[x]}{\sigma_x}
$$

This transforms the dataset into one with zero mean and variance of 1, assuming the distribution is normal $67\%$ of the data will lie between $[-1,1]$.

The values outside this range exists but are rarer.

****

### **<u>Winsorizing</u>**

Uses the **empirical distribution** in the training dataset to clip the dataset bounds given by the 10th and 90th (or 95th and 5th), then applies min-max scale.

****

## Non-Linear Scaling

If the data is skewed and not distributed as uniform or bell curved.

We sometimes apply a non linear transformation before the scaling.

- The non-linear transformation could be 
  - logarithm
  - polynomial expansions(square, sqrt, cube, cubic root ...etc)
  - sigmoid

- We know that we have a good transformation function, if the distribution of the transformed value becomes uniform or normally distributed.

****

You can also do other transforms.

### <u>Histogram Equalization</u>

### <u>Box-Cox transform</u>

****

## <u>Array of numbers</u> 

If the size of the array is constant, It can be flattened and each element can be treated as an input.

If size is not constant:

- Represent array in terms of its bulk statistics, we can use the length, average, median, min, max ...etc
- Represent the array by the nth percentile of its distribution
- If array is ordered by a specific way (time or size), you could take the last n elements for example. (if less than n, pad it with missing vals)

****

## Categorical Inputs

Ordinal representation of the categorical data is not always the way to go.

**One-hot encoding** is done to features.

Sometimes its better to **treat a numeric input as categorical** and map it to a one-hot encoded column:

- Numeric input is an index
  - i.e. days of the week, it is better to do it this way because it is not a continuous representation, as week days start on different days for different countries.
- Relationship between input and label is not continuous
  - Traffic levels on Friday are not affected by those on Thursday and Sunday
- When it is advantageous to bucket the numeric variable
  - As traffic is higher on the weekend, it might be use to categorize days into **weekday** and **weekend**
  - This can also happen to numerical features, as people with close ages are similar to each other, so you can bucket age as such [ <5, 5<x<10, 10<x<15 ... etc.]
- When we want to treat different values of the numeric input as being independent when it comes to their effect on the label
  - If we take baby weight as an example, Triplets are born with lower weight than Twins and twins weight less than single births.
  - So, a lower weight baby, if part of a triplet, might be healthier than a twin baby of the same weight.
  - This can only be done if we have enough examples of twins and triplets.

****

## Array of categorical Input

If previous births of the mother are `[Induced, Induced, Natural, Cesarean]`.

we can 

- Count the occurrences of each vocabulary item, so the array would be `[2, 1, 1]` 
  - To avoid large numbers, we can use **relative frequency** which would be `[0.5, 0.25, 0.25]`.
  - Empty array are represented as `[0, 0, 0]`.
- If an array is a specific order, you could take the last 3 items.
- Represent array by the statistics.

****

# DP1: Hashed Feature

The **Hashed Feature** design pattern addresses **3 problems** associated with **categorical features**.

1. Incomplete vocabulary
2. Model size due to cardinality
3. Cold start

It does so by **grouping the categorical features** and **accepting the trade-off of collisions** in the data representation

****

## Problem

