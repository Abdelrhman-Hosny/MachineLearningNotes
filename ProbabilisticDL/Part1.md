[toc]

# Probabilistic Deep Learning

# Part 1 - DL Basics

# Chapter 1 - Intro to probabilistic models

## 1.1 First Look at Probabilistic Models

### <u>What is a probabilistic model?</u>

Consider the GPS system (satnav) used in everyday life,  the satnav predicts that you move from point A to point B in $x$ minutes, this is not a probabilistic model. 

Here the model predicts a single number $x$ as the time to go from A to B.

A probabilistic model predicts the probability distribution $P(X)$. So it doesn't only capture the travel time but also captures the **uncertainty** of that travel time.

So, if asked to predict the travel time from point A to point B and outputs the two following distributions

![](.\Images\Part1\ProbabilisticModelOutputs.PNG)

This type of prediction allows you to assess the average