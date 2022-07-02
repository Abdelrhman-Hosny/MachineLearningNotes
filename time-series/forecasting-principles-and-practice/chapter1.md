[toc]

# Forecasting

## What can be forecast ?

- Some events can be forecast like future electricity demands in a city
- Other events can't be forecast like the lottery numbers.
- Forecasts can be required a **several years in advance** or only **a few minutes beforehand**.
  - Whatever the circumstances or time horizons involved, forecasting is an important aid to effective and efficient planning.

- Somethings are **easier to forecast** than others.
  - Time of sunrise tomorrow can be forecast precisely.
  - Tomorrow's lotto numbers can't be forecast with any accuracy.

- The **predictability** of an event or a quantity depdends on several factors
  1. Understanding the **factors that contribute to it** 
  2. Availability of **data** 
  3. **Similarity** between the future and the past
  4. Whether the forecast can affect the thing we are trying to forecast

- One of the reasons forecasting **currency exchange rate** is hard is that we only have 1 of the four requirements, we have a lot of data.
  - We have little understanding of the factors that affect the rates
  - Future may be different to the past in the case of a political or financial crisis
  - If we forecast that the rates will increase/decrease, people will adjust the prices they are willing to pay.
    - In a sense, the exchange rates become their own forecasts.
  - As a result, forecasting whether the rates will increase tomorrow is about as predictable as a coin toss.

----------

- In forecasting, a key step is **knowing when something can be forecast accurately**, and when forecasts will be **no better than tossing a coin**.
- Good forecasts should capture the genuine patterns and relationships that exist in the historical data.
  - All while avoiding **replicating past events that won't occur again**.
  - We will learn the difference in this book.

----------

- Many people **wrongly assume** that forecasts **are not possible in a changing environment.** 
  - Every environment is changing and a good model captures the **way in which things are changing**
- Forecasts **rarely assume that the environment is unchanging**
  - They do assume that the **way in which the environment is changing** will continue in the future.
    - e.g. highly volatile envs will continue to be highly volatile, business with fluctuating sales will continue to have them ...etc

- Forecasting models are intended to capture **the way things move** not just where things are.

----------

- Forecastting situations vary widely in their **time horizons, factors determining actual outcome, types of data patterns ..etc**.
- Forecasting methods can be simple or complex

----------

## Forecasting, goals and planning

### Forecasting
- Predicting the future as accurately as possible, given all of the information available.

### Goals
- What you would like to happen.

### Planning
- Response to forecasts and goals.
- Involves determining the appropriate action required to make your forecast match your goals.

----------
## Forecast types

### Short-term forecasts

- Needed for scheduling of personnel, production and transportation.

### Medium-term forecasts
- Needed to determine future resource requirements to 
  - purchase new materials
  - hire personnel
  - buy machinery or equipment

### Long-term forecasts
- Used in strategic planning.
  - e.g. decisions that take acount of markey oppurtunities, environmental factors and internal resources

----------

## Determining what to forecast.

- At the start of each project, you have to ask yourself
  1. What do you need to forecast ?
  2. Time horizon of forecast.
    - Will it be required 1 month in advance, or 6 month in advance ...etc.
    - Different horizons might lead to different model choices.
  3. How often are forecasts required ?

----------

## Forecasting data and methods

- The appropriate forecasting methods depend largely on **what data are available**.

- If little or no data is available, then **qualitative forecasting methods** must be used.
  - Theses are not purely guess work, they are structured to make predictions w/o historical data.

### Quantitative Forecasting

- It can be applied when **two conditions are satisfied**

  1. Numerical information about the past is available.
  2. Reasonable to assume that some aspects of the past patterns will continue into the future.

- There is a wide range of quantitative forecasting  methods
  - Each method has its own properties, accuracies and costs.
- Most quantitative prediction problems use either
  - Time series data
  - Cross sectional data (collected at a single point in time).

----------

## Time Series Forecasting

- Examples
  - annual profits
  - quarterly sales
  - daily stock prices
  - hourly electricity demand
  - time stamped stock transaction data

- Anything observed **sequentially over time** is a **time series**
  - We will only consider time series that are observed at **regular intervals of time**
  - Irregular spaced time series can also occur but they are out of scope of the book.

----------

- When forecasting time series data, the aim is to **estimate how the sequence of observations will continue into the future**

![](https://otexts.com/fpp3/fpp_files/figure-html/beer-1.png)

- The blue lines show forecasts for the next two years.
- Notice how the forecasts **captured the seasonal pattern** seen in the historical data and replicated it for the next 2 years.
  - Also notice the confidence intervals around the predictions.

- The simplest time series forecasting methods use only **information** on the **variable to be forecast**, and make **no attempt** to discover the **factors that affect its behavior** 
  - They will **extrapolate trend and seasonal patterns**
  - However, they'll **ignore** all other information such as **marketing initiatives, competitor activity changes in economic conditions .. etc**.

- **Decomposition methods** are helpful for studying the **trand and seasonal patterns** in a time series.
- Popular time series models used for forecasting include **exponential smoothing and ARIMA models**.

----------

## Predictor variables and time series forecasting

- Predictor variables are often useful in time series forecasting.
- If we wish to forecast the hourly electricity demand (ED) of a hot region during the summer period
  - A model with predictor variables might be of the form.
    `ED = f(current_temp, economy_strength, population, time_of_day, day_of_week, error)`

- We call this an **explanatory model** because it helps explain what causes variability in the demand.

- Because electricity demand data form a time series, we could also use a time series model for forecasting.
  - It would have the form of
    $$
      ED_{t+1} = f(ED_t, ED_{t-1}, ED_{t-2},...,\text{error})
    $$
  - where $t$ is the present hour, $t+1$ is the next hour ...etc
  - The prediction is based on past values of the variable only.

- There is also a third type, which combines both models.
    $$
      ED_{t+1} = f(ED_t, \text{current temperature, time of day, \text{error})
    $$
    - These "mixed models" have been given various names in different disciplines.
      - dynamic regression models
      - panel data models
      - longitudinal models
      - linear system models

- An explanatory model is useful as it incorporates information about other variables rather than only historical values of the variable to be forecase.
- However, there are several reasons to choose a time series model rather than explanatory or mixed model.
  - System may not be understood (or difficult to measure the relationships that are assumed to goven its behavior)
  - necessary to know or forecast future values of the various predictors in order to be able to forecast the variable of interest
    - Collecting that data may prove to be difficult
  - The main concern may be to predict what happens w/o caring why it happens.
  - Time series model may give more accurate forecasts

----------

## The basic steps in a forecasting task

- Forecasting tasks usually involves 5 basic steps.

### 1. Problem definition

- Most difficult part.
- Required understanding
  1. The way the **forecast will be used** 
  2. **Who** requires the forecast
  3. How the forecasting function **fits withing the organisation** requiring the forecast.

- To achieve the understanding required, the forecaster needs to talk to everyone who'll be involved in data collection, database maintenance and using the forecasts for future planning.

----------

### 2. Gathering information

- There are **atleast two** types of information required
  1. Statistical data
  2. Accumulated expertise of the people who collect the data and use forecasts.

- It'll often be difficult to obtain enough historical data to be able to fit a good statistical model.
  - In that case, the **judgemental forecasting  methods** can be used

- Sometimes, we'll get rid of old data as it becomes no longer relevant.

----------

### 3. Preliminary analysis

- Always start by plotting the data.
- Check if there's a
  - significant trend
  - seasonality
  - presence of a business cycle
  - outliers
  - relationships among variables available for analysis

----------

### 4. Choosing and fitting models

- The best model to use depends on 
  1. the availability of historical data
  2. the strength of relationships between the forecast variable and any explanatory variables  
  3. the way in which the forecasts are to be used

- Each model comes with a set of assumptions (explicit and implicit).

----------

### 5. Forecasting model evaluation

- After selection, we use the model to make forecasts.
- We can only evaluate after the dat for the forecast period have become available.
- Some problem arise in **practice**
  1. Handling missing values and outliers
  2. Dealing with short time series

----------






