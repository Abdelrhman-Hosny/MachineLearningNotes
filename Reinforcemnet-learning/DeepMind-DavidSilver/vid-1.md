[toc]

# **<u>Video 1 - Introduction</u>**

## **<u>Characteristics of RL</u>**

- What makes RL different from other machine learning paradigms ?
  1. No supervisor, only a **reward signal**.
  2. Feedback is **delayed**. not instantaneous
  3. Time really matters (input data is **sequential**, non i.i.d data)
  4. Agent's action affect the subsequent data it receives

****

# **<u>The Reinforcement Learning Problem</u>**

## **<u>Rewards</u>**

- A reward $R_t$ is a **scalar** feedback signal
  - It has to be a **scalar** as we want to maximize the reward
- $R_t$ indicates how well the agent is doing at step $t$
- The agent's job is to maximize **cumulative** reward.

#### **<u>Reward Hypothesis</u>**

- All goals can be described by the **maximization of expected cumulative reward**.

#### **<u>Sequential Decision Making</u>**

- **<u>Goal</u>**: select actions to maximize total **future reward**.
- Actions may have long term consequences
- Reward can be **delayed**.
  - Sometimes it's better to sacrifice an immediate reward to gain more reward at the end.
    - Can't be greedy.

****

### **<u>Agent and Environment</u>**

![](./Images/v1/agent-env.png)

- The agent (brain) takes an observation and outputs an action
  - The only thing we can influence is the action.

****

### **<u>History & State</u>**

- The **history** is the **sequence of observations, actions and rewards**
  $$
  H_t = A_1,O_1,R_1,....,A_t,O_t,R_t
  $$
  i.e. all **observable variables** up to time $t$

- Our algorithm will work to turn the history $H_t$ to generate an action $A_{t+1}$.

- But the $H_t$ contains very large amounts of data, most of that data isn't really relevant to the decision of the agent

  - e.g. a helicopter's history might be it's position for the last ten minutes, when only its current position is needed for the decision

  - This is why we use **states**

    - A state is any **function of the history**
      $$
      S_t = f(H_t)
      $$

#### <u>**Types of states**</u>

1. ##### **<u>Environment state</u>**

   - $S_t^e$ is called the **environment state**
   - $S_t^e$ is the environment's **private** representation
     - Whatever data the environment uses to pick the next **observation/reward**.
   - The environment state is not usually visible to the agent.
     - e.g. a tree out of sight of a helicopter is still in the env state but the agent can't see it
   - Even if the agent could see the information, it may contain irrelevant information

   ****

2. **<u>Agent State</u>**

   - $S^a_t$ is the **agent state**.
   - The agent state $S_t^a$ is the agent's **internal representation**
     - i.e. whatever info the agent uses to pick the next action

   ****

3. **<u>Information State</u>**

   - an **information state** (a.k.a. **Markov State**) contains **all useful information from the history**
     $$
     S_t \text{ is a Markov state iff} \\
     P[S_{t+1}|S_t] = P[S_{t+1}| S_1,..,S_t]
     $$
     i.e. probability of the next state given $S_t$ is the same as probability of the next state given all previous states.

   - The future is independent of the past **given the present**.

   - The state is a **sufficient statistic** of the future, and once that state is known, the history may be thrown away.

   - $H_t$ is a Markov state, but not a very useful one as it keeps all of the data.

   - The environment state $S_t^e$ is Markov, as it contains everything.

     - So we can see that it's easy to get Markov States, the hard part is finding compact ones that minimizes the data we need to store.

****

### **<u>Types of environments</u>**

#### **<u>Fully Observable Environment</u>**

- This is the "nice case"

- The agent sees everything

  i.e. $O_t = S_t^e = S_t^a$

  environment state = agent state = information state

- Formally: this is known as a **Markov decision process** (MDP)

****

#### **<u>Partially Observable Environments</u>**

- Partial observability: agent **indirectly observes** the environment

  - A poker playing agent only observes "public" cards

- Now agent state $\neq$ environment state

- Formally: this is know as a **partially observable MDP** (POMDP)

- The agent **must construct its own state representation** $S_t^a$.

- There are several approaches for building your own representations like

  - **Complete History**
    $$
    S_t^a = H_t
    $$
    

  - **Beliefs** of the environment state

    You keep a probability distribution of **which state are you probably in**, and you take the decision based on the probability of states.
    $$
    S_t^a = (P[S_t^a=s^1],...,P[S^e_t = s^n])
    $$

  - **Recurrent Neural Network**
    $$
    S_t^a = \sigma(s_{t-1}^aW_S+O_tW_o)
    $$

****

# **<u>Inside an RL Agent</u>**

- An RL agent may include **one or more** of these components
  1. **<u>Policy</u>**: agent's **behaviour** function
  2. **<u>Value function</u>**: how **good** is each state/action
  3. **<u>Model</u>**: agent's representation of the environment

****

### **<u>Policy</u>**



****

### **<u>Value function</u>**
