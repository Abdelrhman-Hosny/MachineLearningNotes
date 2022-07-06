[toc]

- Time series data can exhibit a variety of patterns, it is often helpful to split a time series into **several components**, each representing an **underlying pattern category**
- We discussed 3 types of time series patterns
  - trend
  - seasonality
  - cycles

- We decompose a time series into components
  - The **trend and cycle** into a single **trend-cycle** component (called the **trend** for simplicity)
  - A **seasonal** component
  - A **remainder** component (containing anything else in the time series)

- For some time series (e.g. those that are observed at least daily), there can be more than **one seasonal component**, corresponding to the different seasonal periods.

- We consider the most common methods for extracting these components of the time series.

----------

## Transformations and adjustments

- Adjusting the historical data can lead to a simpler time series, there are **four kinds** of adjustments in the book
  - calendar adjustments
  - population adjustments
  - inflation adjustments
  - mathematical transformations

- The purpose of these adjustments is to simplify the patterns in the historical data by **removing sources of variation** or **making the pattern more consistent across the whole dataset** 
- Simple patterns are easier to model and lead to more accuracte forecasts


### Calendar adjustments

- Some of the variation seen in seasonal data may be due to simple calendar calendar effects.
  - In such cases, it usually easier to remove the variation before doing any further analysis

- If you study a total monthly sales in a retail store, there will be variation between the months simply because **different months have different number of trading days** in each month.
  - In addition to the seasonal variation across the year.

- It is easy to remove this variation by computing average sales per trading day in each month, rather than total sales in the month.
- This effectively removes the calendar variation.

----------

### Population adjustments

- Any data that is affected by population changes can be adjusted to give per-capita data.
  - Considering the data per person ( per thousand people, or per million people) rather than total.

- This shows whether the change was due to an increase in the demand/use or just due to increase in population.

### Inflation adjustments

- Data affected by the value of money are best adjusted before modelling.
  - House costing 200k in 1990 is not the same as a house costing 200k now.

- To make these adjustments, a **price index** is used.
  - If $z_t$ denotes the price index (measure of relative price change) and $y_t$ denotes the original house price in year $t$
  - $x_t= \frac{y_t}{z_t} * z_{2000}$ gives the adjusted house price at year 2000 dollar values
  - Price indexes are often constructed by government agencies

![](https://otexts.com/fpp3/fpp_files/figure-html/printretail-1.png) 

- By adjusting the price according to the CPI
  - We can see that book retailing industry has been in decline much longer than the original data suggests

### Mathematical transformations

- If the dataset is not **stationary** (changing mean and variance change over time)
- We can apply a mathematical transformation to change the data in order to make it linear (or easier to model in general).

----------

## Time series components

- If we assume an **additive decomposition**, we can write
  $$
    y_t = S_t + T_t + R_t
  $$
  - where $y_t$ is the data, $S_t$ is the seasonal component, $T_t$ is the trend-cycle, and $R_t$ is the remainder component at a period $t$.

- A multiplicative decomposition would be written as $y_t = S_t \times T_t \times R_t$

- The additive decomposition is the most appropriate if
  - The magnitude of the seasonal fluctuations, or the variation around the trend-cycle appears to be proportional to the level of the time series.
  - Otherwise, Multiplicative decomposition is more appropriate.

![](https://otexts.com/fpp3/fpp_files/figure-html/emplstl-1.png) 

- You can use this to remove seasonality from the data.

----------

### Moving average

- We use the moving average to compute the trend-cycle part of the decomposition.

----------

### Computing the decomposition

- First step is to get the trend cycle
  - if $N$ is even compute the $2 \time m-MA$ else compute the $m-MA$

- Get the **de-trended data** $d_t = y_t - t_t$ detrended = data - trend-cycle

- Get the seasonal data $s_t$, where $s_t$ is the average of all de-trended data at the same season.
  - i.e. seasonal data in march is the average of all data points in march

- Lastly, get the remainder $r_t = y_t - s_t - t_t$

