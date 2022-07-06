[toc]

# The Forecaster's toolbox

- This chapter will describe some benchmakr forecasting methods and procedures for checking whether a forecasting method has adequately utilised the available information.

## Tidy forecasting workflow

- The process of producing forecasts for time series data can be broken down into a few steps

  ![](https://otexts.com/fpp3/fpp_files/figure-html/workflow-1.png)
  - We will fit linear trend models to national GDP data stored in `global_economy`

### Data Preparation (Tidy)

- The first step in forecasting is to **prepare** data in the **correct format**
  - This may involve loading data, handling missing values, filtering the time series, and other pre-processing tasks

- Many **models** have **different data requirements** 
  - Some require the series to be in **time order** 
  - Others require **no missing values**

- Checking the data is an essential step to understand its features.

### Plot the data (Visualise)

- Discussed in chapter 2

### Define a model (Specify)

- Different models can be used and specifying an **appropriate model** for the **data** is essential for **producing appropriate forecasts**.

### Train the model (Estimate)

- After choosing an appropriate model, the next step is to train it

### Check model performance (Evalute)

- Similar to ML

### Produce Forecasts (Forecast)

- Similar to ML

----------

## Simple forecasting methods

- We'll define 4 simple methods to forecast time series
- **Sometimes** one of these simple methods will be the **best forecasting method available**
  - But in many cases, these methods will serve as a benchmark rather than the method of choice. (baseline models)

### Mean Method

- The forecast of all future values are equal to the **mean of the historical data** 
  $$
    \hat y_{T+h|h} = \bar y = \frac{y_1 + .... + y_T}{T}
  $$
  - The notation $\hat y_{T+h|h}$ is short hand for estimating $y_{T+h}$ based on the data $y_1,....,y_T$

### Naive method

- We set the forecast to be the value of the **last observation** 
  $$
    \hat y_{T+h|h} = y_T
  $$
 
  - This method works remarkably well for many economic and financial time series

  ![](https://otexts.com/fpp3/fpp_files/figure-html/naive-method-explained-1.png)
    - Naive forecast applied to clay brick production in Australia

### Seasonal naive method

- Useful for **highly seasonal data** 

- We set each forecast to be equal to the **last observed value from the same season** 
  - Jan 2021 = Jan 2022
  $$
    \hat y_{T+h|h} = y_{T + h - m(k+1)}
  $$
    - Where $m$ is the seasonal period, and $k$ is the integer part of $\frac{h-1}{m}$ (i.e. the number of complete years in the forecast period prior to time $T+h$)
 
### Drift Method

- A variation of the naive method that allows the forecast to **change over time**
  - The amount of change over time is called **drift**
  - drift is set to be the **average change seen in historical data** 
  $$
    \hat y_{T+h|h} = y_T + \frac{h}{T-1} \sum^T_{t=2} (y_t - y_{t-1}) = y_T + h (\frac{y_T - y_1}{T-1})
  $$
    - This is similar to drawing a line between the first and last observation and extrapolating into the future.

### Example on simple methods

![](https://otexts.com/fpp3/fpp_files/figure-html/beerf-1.png) 

- Only seasonal naive did well

----------

## Fitted values and Residuals

### Fitted values

- Each observation in a time series can be forecast using all previous observations
  - We call these **fitted values** denoted by $\hat y_{t|t-1}$ or just $\hat y_t$ for simplicity

- Fitted values almost always involve **one-step forecasts**.

- Not all fitted values are forecasts.
  - If an examples requires future data, then not true forecast.

- So a model that assigns the mean of all values to all forecast is a fitted value that is not a true forecast.

----------

### Residuals

- Residuals in a time series model are what is left over after fitting a model
  $$
    e_t = y_t - \hat y_t
  $$

- If a transformation has been used in the model, it is often useful to look at the residuals in the **transformed scale** 
  - We call these **innovation residuals** 
  - For example, if we modelled the data as follows $w_t = \log y_t$, then the **innovation residuals** are given by $w_t - \hat w_t$, while the regular residuals are given by $y_t - \hat y_t$
   
- Residuals are useful in checking whether a model has adequately captured the information in the data.

- If patterns are observable in the innovation residuals, then the model can probably be improved
  - Discussed in the next point : Residual diagnostics

----------

### Residual diagnostics

- A good forecasting method will yield innovation residuals with the following properties
  1. The innovation residuals are **uncorrelated** 
    - If there are correlations between innovation residuals, then there is information left in the residuals which should be used in computing forecasts.
  2. The innovation residuals have **zero mean** 
    - Any mean other than zero means that the forecasts are biased

- Any model that doesn't have one of these **can be improved**.
  - However, that doesn't mean that model that satisfy these properties cannot be improved.

- Using these properties is not a good way to do **model selection**
  - As multiple models can satisfy these properties on the same dataset.

- To correct the non zero mean, simply subtract it from the model by adding a constant $c$
- Correcting the zero correlation problem is harder and will be discussed later on.

----------

- In addition to these **essential** properties, it is **useful (but not necessary)** for the residuals to also have the following two proprties
  3. The **innovation** residuals have **constant variance**  
    - Known as **homoscedasticity** 
  4. The **innovation** residuals are **normally distributed** 

- These two properties make the calculation of prediction intervals easier (examples later)
  - However, a forecasting model that **doesn't satisfy** these properties **cannot necessarily be improved**.

- Sometimes applying a **Box-Cox transformation** may assist with these properties, but otherwise there is **usually little that you can do** to ensure that the innovation residuals have constant variance and a normal distribution

- It'll  be shown later how to deal with non-normal innovation residuals

----------

### Portmanteau tests for autocorrelation

- In addition to looking at the ACF plot, we can do more formal  tests for autocorrelation by considering a whole set of $r_k$ values as a **group** instead of treating each one separately

- Recall that $r_k$ is the autocorrelation for lag $k$, sometimes the ACF will say that there is correlations when there is not (due to the 95% confidence interval)
- To overcome this problem, we test whether the first $l$ autocorrelations are significantly different from what would be expected from a **white noise process** 

- A test for a group of autocorrelations is called a **portmanteau test**

#### Box-Pierce test
- One such test is the **Box-Pierce test**

  $$
    Q = T \sum^l_{k = 1} r^2_k
  $$
  - $l$ is the max lag being considered, and $T$ is the number of observations.

- If each $r_k$ is close to zero, then $Q$ will be small
  - If some $r_k$ values are large, $Q$ will be large too.

- The suggested value of $l$ 
  - $l = 10$ for non-seasonal data
  - $l = 2m$ for seasonal data
    - where $m$ is the period of seasonality
    - However, the test isnt good when $l > T/5$, so if $l > T/5$ clip it so it becomes $l = T / 5$

#### Ljung-Box test

- More accurate than **Box-Pierce test** 

  $$
    Q^* = T(T+2) \sum^l_{k=1} \frac{r^2_k}{T-k}
  $$

- Large values of $Q^*$ suggests that the autocorrelations don't come from a white noise series.

----------

- How large is too large ?
  - If autocorrelations did come from a white noise, then both $Q$ and $Q^*$ would have a $\mathcal{X}^2$ distribution with $(l - K)$ degrees of freedom.
    - $K$ is the number of paramters of the model
    - If the paramters are calculated from raw data (rather than the residuals), then set $K = 0$.

----------






