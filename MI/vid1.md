[toc]

# **<u>CS188 - Introduction</u>**

- The modern view of AI nowadays is trying to make agents that **act rationally**

  - what does **rationality** mean in the world of AI?

    **<u>Rational</u>**: **maximally** achieving predefined goals or **maximizing** your **expected utility**

  - Rationality only concerns the **decisions** not the thoughts behind them.

  - Goals are expressed in terms of the **utility of the outcome**.

****

The Berkeley lectures don't cover chapter 2, that talks about agents, however, it is covered in lectures, so I'll just summarize it from the slides

****

# **<u>University Lecture 2 - Agents</u>**

## **<u>Agents</u>**

- An agent is **anything** that

  - **perceives** its **environment** through **sensors**
  - **acts** upon that environment using **actuators**

  ![](./images/v1/agent.png)

- An agent can be a **software program**.

****

### **<u>Agent function and agent program</u>**

- The **agent function** maps from **percept histories** to **actions**.
- The **agent program** **implements** the agent function to run on a **physical architecture**.
- It is important to keep the two ideas distinct
  - The **agent function** is an **abstract** mathematical description.
  - The **agent program** is a **concrete implementation**, running within some physical system.

****

### **<u>Rationality</u>**

- Rationality depends on

  1. Performance measure
  2. Agent's (prior) knowledge
  3. Agent's percepts to date
  4. Available actions

- **Rational Agent** definition

  For each possible **percept sequence**, a rational agent should select an **action** that is **expected to maximize the performance measure**, given the evidence provided from the **percept sequence** and whatever **build-in knowledge** the agent has.

- We take the **expectation** of the utility due to **uncertainty** in the **environment** (stochastic and partially-observable)

****

#### <u>**Rationality vs Omniscience**</u>

- Rational $\neq$ omniscient
- Omniscient means knows everything
- Rationality means maximizing the **expected** performance, while perfection (omniscient) maximizes the **actual** performance.

****