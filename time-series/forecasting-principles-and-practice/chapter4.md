[toc]

# Time series features

## Simple statistics

- It's common to compute some statistics for the time series.
  - Max, min
  - mean, median
  - Quantiles

----------

## ACF features


- All the autocorrelations can be considered features for that series.
- We can also summarise the autocorrelations to produce new features, like
  - sum of the first ten squared coefficients
  - autocorrelations of the changes in the series between periods
    - difference between data
  - ACF of the double difference
  - ACF of seasonally differenced series
    - e.g. create a series that consists of the difference between the time series each sunday and get the ACF

----------

### STL Features

- Recall that a time series can be decomposed into
  $$
    y_t = S_t + T_t  + R_t
  $$

- You can use these to get other features, e.g.
  - For strongly trended data, the seasonally adjusted data should have much more variation than the remainder component
    - Which leads to a relatively small $\frac{Var(R_t)}{Var( T_t + R_t)}$
    - However, if the data has little or no trend, the two variances should approximately be the same.
  - This allows use to define the **strength of trend** as
    $$
      F_T = \max (0, 1 - \frac{Var (R_t)}{Var (T_t + R_t)})
    $$

  - The same concept is applied to get the **strength of seasonality**
    $$
      F_s = \max ( 0, 1 - \frac{Var ( R_t)}{Var (S_t + R_t)})
    $$

- These measures can be useful when you have a lot of time series and want to know who has the **highest trend** or **highest seasonality**

### Other features

- There are other features
  - Hurst coefficient
  - Box pierce
  - Ljung-Boxx statistic
