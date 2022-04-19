[toc]

<div style='page-break-after: always;'></div>

# <u>**Types of Games**</u>

### Classification

- Deterministc, Stochastic
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
  - Termnal Test: $S \rightarrow \{t,f\}$
    - Is the game over or not in the current situation?
  - Terminal Utilities: $S\times P \rightarrow R$
    - Every state has a **score**
    - Simple games may have only 3 states { Win, Lose, Draw } while others may have more than that

- Solution for a **player** is a **policy** $S \rightarrow A$

****

## <u>**Zero-sum games**</u>

|Zero-Sum Games | General Games|
|:-------------|:-------------|
|Agents have opposite utilities| Agents have **independent** utilities|
|If one agent maximizes the utility, the other's best option is to **minimize** the **same utility** | Co-operation, indifference and competion are all possible|
|Adversarial, pure competition||

****
