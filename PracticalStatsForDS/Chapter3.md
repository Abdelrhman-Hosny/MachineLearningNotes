**<u>Chapter 3</u>**

# **<u>Statistical Experiments and Significance Testing</u>**

Designing experiments is a cornerstone of the practice of statistics. The goal is to design an experiment in order to confirm or reject a hypothesis.

This chapter reviews traditional experimental design and discusses some common challenges in data science. It also covers some often cited concepts in statistical inference and explains their meaning and relevance or lack thereof to data science.

Statistical significance is typically seen in the context of the classical statistical inference. The process goes as follows :

1. Formulate a hypothesis i.e. "Drug A is better than the existing standard drug"
2. An experiment is designed to test the hypothesis
3. Data is collected and analyzed
4. A conclusion is drawn

<u>**Inference**</u>: reflects the intention to apply the experiment results , which involve a limited set of data, to a larger process or population.

------

## <u>**A/B Testing**</u>

An A/B test is an experiment with two groups to establish which of the two is superior.

The existing thing that is being used is called the **control**. A typical hypothesis is the something is better than control.

| Term                | Definition                                                   |
| ------------------- | ------------------------------------------------------------ |
| **Treatment**       | Something(drug,price,web headline) to which a subject is exposed |
| **Treatment group** | A group of subjects exposed to a specific treatment          |
| **Control group**   | A group of subjects exposed to no (or standard) treatment    |
| **Randomization**   | The process of randomly assigning subjects to treatments     |
| **Subjects**        | The items (web visitors,patients ... etc.) that are exposed to treatments |
| **Test statistic**  | The metric used to measure the effect of the treatment       |

A/B tests are common in marketing and web design as results are readily measures. Some A/B tests include: 

- Testing two soil treatments to determine which produces better seed germination
- Testing two prices to determine which yields more net profit
- Testing two web headlines to determine which produces more clicks

A proper A/B test has *subjects* that can be assigned to one treatment or another. The subject might be a person , a plant ... etc ; the key is that the subject is exposed to the treatment. Ideally, subjects are *randomized* ( assigned randomly ) to treatments.

In this way , you know that any difference between the treatment groups is due to one of two things:

- The effect of the different treatments
- Luck of the draw in which subjects are assigned to which treatment (i.e. , Drug B got tested on healthier people than the ones whom drug A was tested. )

You also need to pay attention to the *test statistic*. Perhaps the most common metric in data science is a binary variable.

Example on experiment results on test prices.(Binary Variable)

|    Outcome    | Price A | Price b |
| :-----------: | :-----: | :-----: |
|  Conversion   |   200   |   182   |
| No conversion | 23,539  | 22,406  |

Example on experiment with prices. ( Continuous variable )

| Revenue/page-view | Mean | std_dev |
| :---------------: | :--: | :-----: |
|      Price A      | 3.87 |  51.1   |
|      Price B      | 4.11 |  62.98  |

**Keep in mind : **Just because values are generated doesn't mean they have meaning , the values of std dev imply that there could be negative revenue. This is due to that a lot of people will go to the page and not pay anything and other will go and pay hundreds which makes these statistics not the best way to measure.

------

#### <u>**Why have a Control Group ?**</u>

If A is the control group , why do we re-test when we could take previous experiences?

We re-measure things to make sure that the same conditions are applied to A and B so that the only errors are from differences and sampling

**N.B.** Researchers have to decide on a *test statistic* before starting the experiment. Setting the statistic after the experiment is conducted opens the door to *researcher bias*

------

## **<u>Hypothesis Tests</u>**

Also called *significance tests* , are ubiquitous in the traditional statistical analysis of published research. Their purpose is to **help you learn whether random chance might be responsible for an observed effect**

| Term                       | Definition                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **Null hypothesis**        | The hypothesis that chance is to blame                       |
| **Alternative hypothesis** | Counterpoint to the null ( what you hope to prove )          |
| **One-way test**           | Hypothesis test that counts chance results only in one direction |
| **Two-way test**           | Hypothesis test that counts chance results in two directions |

Statistical hypothesis testing was invented as a way to protect researchers from being fooled by random chance ( which happens a lot as humans underestimate the scope of natural random behavior).

------

### **<u>Misinterpreting Randomness</u>**

if you ask a human to invent a series of 50 coin flips , and then after that have them actually flip a coin 50 times and find the answer.

You'll find that it is easy to tell which results are real , the real ones will have longer runs of Hs or Ts . as 5 or six Heads or Tails in a row isn't that unusual. However , when inventing if you write 3 or 4 in a row , we tell ourselves that , for the series to look random we better switch it up.

The other side of this , is that when we see the real-world equivalent of six Hs in a row (i.e a headline outperforming the other by 10%) , we are inclined to attribute it to something real , not just a chance.

------

In a properly design A/B test , you collect data on treatments A and B in a way that any observed difference is due to :

- Random chance in assignment of subjects
- A true difference between A and B

A statistical hypothesis test is further analysis of an A/B test , or **any randomized experiment** , to asses whether random chance is a reasonable explanation for the observed difference between A and B.

### **<u>The Null Hypothesis</u>**

On A/B test ,  this assumes that both treatments are equivalent and any difference between them is due to chance. This baseline assumption is termed the **null hypothesis**.

Our hope is that we can prove the null hypothesis wrong , and show that outcomes for groups A and B are more different than what chance might produce.

One way to do this is via a resampling permutation procedure , in which we shuffle together the results from groups A and B and repeatedly deal out the data in groups of similar sizes , then observe how often we get a difference as extreme as the observed difference.

### **<u>Alternative Hypothesis</u>**

Hypothesis tests by nature involve not just a null hypothesis , but also an offsetting **alternative hypothesis**. Examples :

- Null = "A and B are equivalent". Alt = "A and B are different"
- Null = "A $\le$ B", Alt = "B > A"
- Null = "B is not greater than A " , Alt = "B  is greater than A"

Notice that the null and alternative hypothesis must account for all possibilities. The nature of the null hypothesis determines the structure of the hypothesis test.

### <u>**One-Way, Two-Way Hypothesis Test**</u>

A one-tail hypothesis means you have a one directional alternative hypothesis (B is better than A) this means that the extreme chance results in only one  direction count towards the p-value , while a two-tail hypothesis means you have a bidirectional alternative hypothesis ( A is different than B ) this means that the extreme chance results in either direction count towards the p-value

------

## **<u>Resampling</u>**

There are two types of sampling procedures : the **bootstrap** and **permutation tests**.

The bootstrap : used to assess the reliability of an estimate

Permutation tests : used to test hypotheses , typically involving two or more groups (discussed here)

| Term                 | Definition                                                   |
| -------------------- | ------------------------------------------------------------ |
| **Permutation Test** | The procedure of combining two or more samples together , and randomly (or exhaustively) reallocating the observations to resamples |

------

### <u>**Permutation Test**</u>

In a permutation procedure , two or more samples are involved. The first step in a *permutation* test is to combine the results from groups A and B (all the data if there are others) together. This is the logical embodiment of the null hypothesis that the treatments to which the groups were exposed do not differ. We then test that hypothesis by randomly drawing groups from this combined set , seeing how much they differ from one another.

#### **<u>Permutation Procedure</u>**

1. Combine the results from the different groups is a single data set
2. Shuffle the combined data , then randomly draw (no replacement) a resample of the same size as group A
3. From the remaining data , randomly draw a resample of the same size as group B
4. Do the same groups C , D ... etc.
5. Whatever statistic or estimate was calculated for the original samples , calculate it now for the resample and record ; this constitutes one permutation iteration
6. Repeat the previous steps $R$ times to yield a permutation distribution of the test statistic

After doing this go back to the observed difference between groups and compare it to the set of permuted differences. If the observed difference lies well within the set of permuted differences , then we have not proven anything.

However , if the observed difference lies outside most of the permutation distribution , then we conclude that *the difference is statistically significant* (i.e. chance is not responsible)

------

#### **<u>Example:  Web Stickiness</u>**

Problem:

A company selling a relatively high-value service wants to test which of two web presentations does a better selling job. Due to the high value of the service , sales are infrequent and the sales cycle is lengthy ; it would take too long to accumulate enough sales to know which presentation is superior.

So the company decides to measure the results with a proxy variable , using the detailed interior page that describes the service

**Proxy variable**: a variable that represents another variable (i.e. height can be used as a proxy for weight as taller people tend to be heavier) . Proxy variables are used because other variables might be hard/expensive to collect so having a representation of it is better than having nothing.

So we use the average session time as our metric for comparing page A to page B.

We compute the mean of our metric on page A and page B $\mu_{\text{diff}}$ . then we perform a permutation test with $R = 1000$  and compute the difference of means again.

One of two cases will happen :

1. The difference of means from the permutation test will be greater than or equal to $\mu_{\text{diff}}$ . In that case this was just random chance.
2. The difference of means from the permutation test will be less than $\mu_{\text{diff}}$ . In that case , depending on whether the difference of means was positive or negative , we say the one page is better than the other.

We don't compare the initial metric with every result from the test , we say that if less than 5% of the test results were greater than it , then there's a difference else it was just the randomness.

------

### **<u>Exhaustive and Bootstrap Permutation Test</u>**

#### **<u>Exhaustive Permutation Test</u>**

This works for only small data sets as instead of randomly sampling we figure out all the possible permutations of the data and test on them instead.

Sometimes called *exact tests*.

#### **<u>Bootstrap Permutation Test</u>**

Same as the normal permutation test but we sample with replacement

------

A virtue of Permutation tests is that it comes closer to a "one size fits all" approach to inference. You can apply it on nearly all types of models and data with no previous assumptions.

------

## **<u>Statistical Significance and p-values</u>**

| Term             | Definition                                                   |
| ---------------- | ------------------------------------------------------------ |
| **P-value**      | Give a chance model that embodies the null hypothesis, the o-value is the probability of obtaining results as unusual or extreme as the observed results |
| **Alpha**        | The probability threshold of "unusualness" that chance results must surpass,for actual outcomes to be deemed statistically significant |
| **Type 1 error** | Mistakenly concluding an effect is real (when it is due to chance) |
| **Type 2 error** | Mistakenly concluding an effect due to chance (when it is real) |

