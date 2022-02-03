[toc]

# **<u>Chapter 1 - Introduction</u>**

- The book starts off by mentioning different terms related to pattern recognition and machine learning

- It also talks about the difference between **training** and **practical applications**

  - As training data will always comprise only a **tiny fraction** of **all possible input vectors**.
  - This means that we have to aim for **generalization** when using ML or Pattern Recognition.

- It also talks about **preprocessing** and how train and test data must have the same processing steps applied to both of them

  - It also talks about how preprocessing might be used to optimize time performance not only accuracy/prediction (e.g. viola jones face detection).

- Mentions the difference between **classification** and **regression**.

- Mentions **supervised**, **unsupervised** and **reinforcement learning**.

- Mentions how **unsupervised learning** can be used

  - Density estimation
  - Clustering
  - Visualization (think PCA allowing us to visualize word embeddings).

- Talk about **reinforcement learning** which I am not as experienced in.

  - Reinforcement learning is concerned with the problem of **finding suitable actions to take in a given situation to <u>maximize a reward</u>**.

  - So, the learning algorithm is not given **examples of optimal outputs** (like in supervised learning). Instead it must **discover them** through a process of **trial and error**.

  - Typically, there is a sequence of **states** and **actions** in which the learning algorithm is **interacting with its environment**.

  - In many cases, the **current action** not only affects immediate rewards but also has **an impact on the rewards of <u>all subsequent time steps</u>**

  - If we take for example, using RL to play backgammon the inputs for the model would be : the board position and the value of the dice throw and the output would be a move.

    A major challenge called the *credit assignment problem* is that backgammon has a lot of moves, and we can only give the reward at the end of the game (when victory is achieved). So the reward has to be **attributed appropriately to <u>all the moves that led to it</u>** even though not all moves played contributed equally to the victory.

  - Another problem is the *trade-off* between **exploration** and **exploitation**.

    **exploitation:** when the system makes use of actions that are **known to give a high reward**.

    **exploration:** when the system tries new kind of actions to check **how effective they are**.

  - If a model focuses too strongly on either exploration or exploitation, the model will yield poor results.

****

## **<u>Polynomial Curve Fitting</u>**

- Suppose we are given a set of inputs $\bf{x} = \{ x_1,x_2,....,x_N\}$ corresponding to the output values $\bf{t} = \{ t_1, t_2, ....,t_N\}$.
- We will assume that $t = sin(2 \pi x) + \mathcal{N}(0, 1)$ where this is a sine plus a Gaussian noise.
- This is similar to what machine learning tries to do in **supervised learning**, we have a function that we are trying to predict which is $sin(2\pi x)$ and some noise.
- We perform polynomial curve fitting using

$$
y(x,\bf{w}) = w_0 + w_1 x+ w_2 x^2 + .... + W_M x^M = \sum_{j=0}^Mw_jx^j
$$

and for the error we use the **RMSE**.

![](./Images/ch1/poly-curve-fitting.png)

We can see that at $M=9$ and $N=10$, the training set error goes to zero but why at $M=9$ precisely ?

- At $M=9$, this polynomial contains **10 degrees of freedom** corresponding to the **10 coefficients**. $w_0,.....,w_9$ and so it can be tuned exactly to the **10 data points** in the training set.
  - This is why we can afford a more complex data model as $n$ (amount of data increases), because at a high $n$, we have a higher degree of freedom, so we can increase $k$ (# of features) without overfitting.

****

## **<u>Probability Theory <3</u>**

- Probability theory naturally arises in pattern recognition through **noise** and **finite size of datasets**.
- Probability theory combined with **decision theory** allow us to make **optimal predictions** based on the information available to us, even though that information may be **incomplete** or **ambiguous**.
- The book introduces some rules of probability

$$
\text{sum rule (marginal probability)} \ : \ P(X) = \sum_Yp(X,Y) \\ \text{product rule} \ : \ P(X,Y) = p(Y|X)p(X) = P(X|Y)p(Y) \\ \text{bayes rule : } P(Y|X) = \frac{P(X|Y) P(Y)}{P(X)} = \frac{P(X|Y) P(Y)}{\sum P(X|Y)P(Y)} \\
$$

****

## **<u>Probability densities</u>**

Check the notes from probability course

****

## **<u>Bayesian Probabilities</u>**

- The probability we reviewed so far was in terms of the **frequencies** of random and **repeatable** events. this is called the **classical** or **frequentist** interpretation of probability.

- What about events that can't be counted ?

  - e.g. the probability that the Arctic ice cap will have disappeared by the end of the century
    - This event can't be repeated for us to study and get the probability

- Nevertheless, we will generally have **some idea** (e.g. rate of melting of the ice) that can help us determine the probability

  - The **some idea** that we have is a reference to the **prior** information
    $$
    P(\theta|data) = \frac{P(data|\theta) P(\theta)}{P(data)}
    $$
    $P(\theta)$ is the prior, $P(data|\theta)$ is the likelihood and $P(\theta|data)$ is the posterior.

- Usually, $P(\theta)$ is not the same, through time its value might change and this will result in a change in our prediction too.

  - e.g. Satellites provide us with new data about the rate of melting of ice.

- **Bayesian probability** allows us to **quantify** our **expression of uncertainty** and change that expression based on **new data and observations**.

  - In turn this will lead us to making better decisions

****

### **<u>Example: Polynomial Curve Fitting</u>**

- If we want to fit a curve, it seems reasonable to apply the **frequentist** method. However, we would like to qunatify the **uncertainty** that surrounds the appropriate choice for the **parameters** $\bf w$ or the **choice of model** itself.
- Recall that, we have previously used **Bayes Theorem** to convert a **prior** probability to a **posterior** probability by **incorporating the evidence obtained by the data**.
  - We will see later that we can adopt a similar approach when making inferences about quantities such as **parameters** in the polynomial curve fitting.

****

- The effect of the observed data $\mathcal{D} = \{ t_1,....,t_N\}$ is expressed through the conditional probability $P(\mathcal{D}| \bf w)$, this can be represented using Bayes Theorem.
  $$
  P(\bf w|\mathcal{D}) = \frac{P(\mathcal{D}|\bf w)P(\bf w)}{P(\mathcal{D})}
  $$
  This allows us to evaluate the **uncertainty** in $\bf w$ after we have observed $\mathcal{D}$ in the form of the **posterior probability** $P(\bf w| \mathcal{D})$.

- $P(\mathcal{D}|\bf w)$ is called the **likelihood function**, it expresses how **probable the observed data** set is for different values of the parameter vector $\bf w$.

- Bayes theorem can be rephrased to words as
  $$
  \text{posterior} \propto \text{likelihood} \times \text{prior}
  $$
  where all the quantities are viewed as **functions of** $\bf w$. we ignore the denominator as it is just a normalizing factor.

- The denominator $P(\mathcal{D})$ can be expressed through the prior and the likelihood function
  $$
  P(\mathcal{D}) = \int P(\mathcal{D}|\bf w)p(\bf w) \dd\bf w 
  $$

- In both the **Bayesian** & **frequentist** paradigms, the likelihood function $P(\mathcal{D}|\bf w)$ plays a **central role**

  - However, the **manner in which it is used** is fundamentally different in the two approaches.

****

### **<u>Likelihood</u>**

- In the frequentist setting, $\bf w$ is considered a **fixed parameter**, whose value is determined by some form of **estimator**.
  - Error bars on this estimate are obtained by considering the distribution of possible data sets $\mathcal{D}$
- By contrast, from the bayesian viewpoint there is only a **single dataset** $\mathcal{D}$ (the one being observed)
  - The uncertainty of the parameters is expressed through a distribution over $\bf w$.
- A widely used **frequentist estimator** is **maximum likelihood**, in which $\bf w$ is set to the value that maximizes the $P(\mathcal{D}|\bf w)$.
  - An approach to determining **frequentist error bars** is bootstrapping.
- One advantage of the **Bayesian viewpoint** is that the **prior information is naturally included**. e.g.
  - Suppose a coin is tossed 3 times and lands head each time.
    - A classical maximum likelihood estimate of probability of landing heads would give one (i.e. $P(Heads) = 1$), implying all future tosses will land heads.
    - However, The Bayesian approach would make use of **prior** and would yield a less extreme result.

****

## **<u>Curve fitting re-visited</u>**

