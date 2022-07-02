[toc]

# Time Series Graphics

## Time Series Patterns

### Trend

- A trend is a **long term increase/decrease** in the data.
- It doesn't have to be linear.
- Sometimes, we'll refer to a trend as **changing direction**, when it goes from an **increasing trend** to a **decreasing trend**.

- Here is an example of an increasing trend
![](https://otexts.com/fpp3/fpp_files/figure-html/a10plot-1.png) 

----------

### Seasonal

- A **seasonal pattern** occurs when a time series is affected by **seasonal factors** such as
  - Time of the year
  - Day of the week

- Just like how sales increase during the holidays/start of year/end of year ...etc

![](https://otexts.com/fpp3/fpp_files/figure-html/a10plot-1.png) 
- In this image, we see that there's always a drop at the end of the year (Dec -> Jan)

----------

### Cyclic

- A cycle occurs when the data exhibit **rises and falls** that are **not of a fixed frequency**
- These fluctuations are usually due to economic conditions, often related to the business cycle.
  - The duration of these fluctuations is usually atleast 2 years.

----------

### Seasonal vs Cyclic

- If fluctuations are **not of a fixed frequency**, they are **cyclic** 
- If the frequency is **unchanging** and **associated with some aspect of the calendar** then the pattern is **seasonal**

- In general, the average length of a cycle is longer than the length of a seasonal pattern. (same applies to the magnitudes)

----------

### Examples

![](https://otexts.com/fpp3/fpp_files/figure-html/fourexamples-1.png) 

1. Top Left
  - No obvious trend in data
  - Shows strong seasonality withing each year
  - Strong cyclic behavior with a period of about 6-10 years

2. Top right
  - Obvious downward trend
  - There is no seasonality.
  - If we had a **longer time series** (more than 100 days), we would see that this downward  trend is actually part of a long cycle.
    - But due to the limited data, we see this as a trend.

3. Bottom Left
  - Upward trend
  - Seasonal behavior
  - No cyclic evidence

4. Bottom right
  - No trend
  - No seasonality
  - No cyclic behavior
  - There doesn't seem to be any pattern in this data.
    - This would make it very hard if not impossible to make a forecasting model.

----------

## Seasonal plots

- The data is plotted against individual seasons.

![](https://otexts.com/fpp3/fpp_files/figure-html/seasonplot1-1.png) 

- Here each line represents a year of sales

### Multiple seasonal plots

- Where the data has more than one seasonal pattern.

- Electricity demand each day (Shows **daily seasonal patterns** )
  ![](https://otexts.com/fpp3/fpp_files/figure-html/multipleseasonplots1-1.png) 

- Electricity demand each week (Shows **weekly seasonal patterns** )
  ![](https://otexts.com/fpp3/fpp_files/figure-html/multipleseasonplots2-1.png) 

- Electricity demand each year (Shows **yearly seasonal patterns** 
  ![](https://otexts.com/fpp3/fpp_files/figure-html/multipleseasonplots3-1.png) 

## Seasonal subseries plots

- An alternative that emphasises **seasonal patterns** where the data for each season are collected together in separate mini time plots

![](https://otexts.com/fpp3/fpp_files/figure-html/subseriesplot-1.png) 

- The blue horizontal line is the mean of each month.
- This type of plot is especially useful in identifying changes within particular seasons

----------

## Scatter plots

- Useful to explore **relationships between time series**

----------

## Lagged plots

- Lagged plots are scatter plots, where each graph shows $y_t$ plotted against $y_{t-k}$ for different values of $k$.

![](https://otexts.com/fpp3/fpp_files/figure-html/beerlagplot-1.png) 

----------

## Autocorrelation

- Autocorrelation measures the **linear relationship between lagged values of a time series** 
- There are several autocorrelation coefficients, corresponding to **each panel in the lag plot**
  - $r_{1}$ measures the relationship between $y_{t}$ and $y_{t-1}$
  - $r_{2}$ measures the relationship between $y_{t}$ and $y_{t-2}$
  - ...etc

- The value of $r_k$ can be written as
  $$
  r_k = \frac{\sum_{t = k + 1}^T (y_t - \bar y)(y_{t-k} - \bar y)}{\sum_{t=1}^T (y_t - \bar y)^2}
  $$
  where $T$ is the **length of the time series**

- The **autocorrelation coefficients** make up the **autocorrelation function (ACF)**

- We usually plot the ACF to see how the **correlations change with the lag** $k$.
  - This plot is sometimes known as a **correlogram**,
  - Example
  ![](https://otexts.com/fpp3/fpp_files/figure-html/beeracf-1.png) 

    - $r_4$ is higher than for the other lags (due to the seasonal pattern in the data tends to be 4 quarters apart)
    - $r_2$ is more negative than for the other lags because troughs tend to be two quarters behind peaks

----------

### Trend and seasonality in ACF plots

- When the data has a **trend**, the autocorrelation for small lags tend to be large and positive
  - As observations nearby in time are also nearby in value.
  - So the ACF of a trended time series tends to have positive values that slowly decrease as the lag increases

- When the data is **seasonal**, the autocorrelations will be **larger** for the **seasonal lags** (at **multiples of the seasonal period** ) than for other lags

![](https://otexts.com/fpp3/fpp_files/figure-html/acfa10-1.png) 
- When data is both trending and seasonal, you see a combination of the two above phenomenas.
  - Similar to the image above

----------

## White noise

- Time series that show **no autocorrelation** are called **white noise** 

- Example
  ![](https://otexts.com/fpp3/fpp_files/figure-html/wnoise-1.png) 

  - ACF for white noise
  ![](https://otexts.com/fpp3/fpp_files/figure-html/wnoiseacf-1.png) 


- For white noise series, we expect each autocorrelation to be **approximately zero**.
  - We expect 95% of the spikes to lie within $\pm \frac{2}{\sqrt{T}}$ where $T$ is the length of the time series.
  - It's common to plot these bounds on the ACF (blue dashed lines above)
  - In the data above $T=50$.
