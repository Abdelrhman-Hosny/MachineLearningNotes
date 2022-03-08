# **<u>Lectures</u>**

- **<u>Reflex Agents</u>**

  Don't consider the **future consequences** of their actions.

  May have memory or a model for the world's current state

  - Reflex agents **can be **rational.

- **<u>Planning agents</u>** (not in uni lecture slides)

  Asks **what if**. Decisions are based on a **hypothesized consequences of actions**.

  Must have a **model of how the world works**.

  **Optimal**: Achieve goal in minimum cost

  **Complete**: If the solution exists, you'll find it

  **<u>Replanning</u>**: Plans only a part of the whole thing, and re-plans after some time

****

## **<u>Search problems</u>**

- A search problem consists of
  1. A state space (set of all possible states that could take place).
  2. A **successor** function (takes an action and predicts the next state of the world along with the **cost of the action**).
  3. A **start state** and a **goal test** (test to determine whether goal is reached).
-  A solution is a **sequence of actions** (plan) which transforms the **start state** to a **goal state**.
- To solve a search problem, you have to **model** it correctly.

****

## **<u>Abstraction</u>**

- Means that we **remove the details**.
  - e.g. If we take the road x, we don't care what radio channel was playing
  - This makes modeling problems easier

****

## **<u>State Space Graphs and Search Trees</u>**

