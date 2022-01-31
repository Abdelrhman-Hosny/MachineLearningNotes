[toc]

# Zed statistics - Degrees of Freedom

[Video Link](https://www.youtube.com/watch?v=N20rl2llHno)

****

## **<u>DF in Descriptive Statistics</u>**

- Consider counting the spikes on a sample of 5 sea urchins (retsa - this is what I call them idk)
  - $x = [213,180,175,242,295]$
- If we calculate some estimates for the data, we get the following

| Statistic        | Value | Degree of freedom |
| ---------------- | ----- | ----------------- |
| Estimate Mean    | 221   | $DF = n = 5$      |
| Mean             |       | $DF = n = 5$      |
| Estimate Std Dev | 49.4  | $DF = n - 1 = 4$  |
| Std Dev          |       | $DF = n = 5$      |

- What is the degree of freedom ?

  - DF are the number of **independent** pieces of information that we have to estimate some value

  - The values we are trying to **estimate** here are mean and std dev

    - Notice that **estimate** is highlighted

      The value that we get from these 5 examples is not the **population mean** or the **population std dev**. But it is the best thing we can get given our data.

- Lets simplify things a bit and assume $x = [213]$ (i.e. we only have 1 sample) and lets get the mean and the variance. (estimates)

  - Think how many independent pieces of information are given to you when calculating
  - $\bar x = 213$, we are given 1 example so our $DF = n = 1$ when estimating mean
  - $var = \frac{\sum_i (x_i - \bar x)^2}{n-1}$.  mathematically, calculation the variance can't happen as we get division by zero, but why does that happen ?
    - variance calculates how far are you from the mean, but at $n = 1$, the estimate of variance for any random variable will be equal to zero.
    - That would be the equivalent of asking how far is the number 2 from 2 which doesn't logically make sense.

- What happens when you have $x = [213, 180]$

  - Estimate mean = $196.5$.
  - Actual variance = $\frac{\sum(x - \mu)^2}{n}$. notice that if we have the population mean $\mu$, we have $n$ degrees of freedom.
  - What if we only have $\bar x$, then we will get the estimate variance which has $n-1$ degrees of freedom as we explained why above.
    - Another thing that $n-1$ does is that since we are dividing by $n-1$ instead of $n$, we are increasing the value of $s^2$.
    - This is done as choosing $\bar x$ for the calculation of $s^2$ gives the minimum value for it given our sample. so dividing by $n-1$ instead of $n$ makes the value closer to the population variance.

****

## **<u>DF in Regression</u>**

- Assume that you are running a regression that estimates the number of spikes on sea urchins based on temperature
  - $spikes_i = \beta_0 + \beta_1 * temp_i + \epsilon_i$ where epsilon is the error in each example.
- How many points are needed so that we can run this regression?
  - while 2 points may sounds intuitive at first, as you can fit a line onto two points, it is a wrong answer.
  - If we only have two points, we can estimate $\beta_0$ and $\beta_1$ but we will always have an error of 0 ($\epsilon_i=0$). So we can't estimate the uncertainty on each element.
  - It is only when you have $n=3$ that you can run this regression and expect meaningful values.
- $DF = n - k - 1$ where $n$ is the number of examples/inputs and $k$ is the number of variables.
  - This is why we can afford a more complex data model as $n$ (amount of data increases), because at a high $n$, we have a higher degree of freedom, so we can increase $k$ (# of features) without overfitting.

****