[toc]

# Dynamic regression models

- So far, we have discussed models that allowed us to use previous target values to predict current target values and others that use different predictors to predict the current target value
  - Each of these were discussed separately, where you had to pick one or the other.

- In this chapter, we will try and combine both.

- This is useful in case you want to include the effect of holidays in ARIMA models for example.

- In this chapter, we will allow the **errors** from a **regression** to **contain autocorrelation**.
  - To emphasise this change in perspective, we will replace $\epsilon_t$ with $\eta_t$ in the equation
    $$
      y_t = \beta_0 + \beta_1 x_{1,t} + .... + \beta_k x_{k,t} + \eta_t \\
      (1 - \phi_1 B) (1 - B) \eta_t = (1 + \theta_1 B) \epsilon_t
    $$
    assuming $\eta_t$ follows an ARIMA$(1,1,1)$ model.

- Notice that the model now has **two error terms**
  - Error from regression denoted by $\eta_t$
  - Error from ARIMA denoted by $\epsilon_t$
  Only the ARIMA model errors are assumed to be white noise.

----------

## Estimation


