[toc]

<div style='page-break-after: always;'></div>

# <u>**Types of Games**</u>

### **<u>Classification</u>**

- Deterministic, Stochastic
- One, Two or more players
- Zero sum ?
- Perfect information (can you see the state)?

### <u>**Solution**</u>

- Solution for this is a **policy** which **recommends a move from each state**.

****

## <u>**Deterministic Games**</u>

- Formulation of the problem:
  - States: $S$ (start at $s_o$)
  - Players $P={p_1, p_2,...,p_n}$ (usually take turns)
  - Actions: $A$ (may depend on player/state)
  - Terminal Test: $S \rightarrow \{t,f\}$
    - Is the game over or not in the current situation?
  - Terminal Utilities: $S\times P \rightarrow R$
    - Every state has a **score**
    - Simple games may have only 3 states { Win, Lose, Draw } while others may have more than that

- Solution for a **player** is a **policy** $S \rightarrow A$

****

## <u>**Zero-sum games**</u>

|Zero-Sum Games | General Games|
|:-------------|:-------------|
|Agents have **opposite** utilities, this allows us to only have 1 number as a measure of performance, where agent one tries to maximize and agent 2 tries to minimize that number| Agents have **independent** utilities|
|If one agent maximizes the utility, the other's best option is to **minimize** the **same utility** | Co-operation, indifference and competition are all possible |
|Adversarial, pure competition||

****
## <u>**Adversarial Search**</u>

- It's a type of search, where you have to take into account the actions of another agent.

### <u>**Minimax Search**</u>

- Works for **deterministic, zero-sum** games
  - Tic-tac-toe, chess, checkers
  - One player maximizes the result
  - The other player minimizes the result

- <u>**Minimax Search**</u>
  - Contains a state-space search tree
  - Players **alternate turns**
  - Compute each node's **minimax value**
    - The best achievable utility against an **optimal** adversary

```python
def minimax(state, player):
    if state.is_terminal:
        return state.utility(player)
    if player == 1:
        return max_value(state)
    else:
        return min_value(state)

```

#### <u>**Minimax Properties**</u>

- Minimax is only optimal if the opponent plays **optimaly**
- **Complexity**
  - Exhaustive search (like DFS)
  - Time: $O(b^m)$
  - Space: $O(bm)$
- **Example**, chess $b \approx 35, m \approx 100$
  - Exact solution is infeasible
  - Improvements ???

