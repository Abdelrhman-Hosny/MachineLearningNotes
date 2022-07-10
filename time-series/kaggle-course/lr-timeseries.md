- We are going to practice doing linear regression with time series data
- We will use two important features
  1. time-step feature
    - Features that can be derived directly from the time index
    - The most basic one is the **time dummy** which counts off time steps in the series from beginning to end.
  2. lag feature
    - The value of the target in the previous timestep (can have more than one timestep if needed).

- **Time step features** let you model **time dependence**.

## Code

### Time step

```python
df = pd.read_csv(....)

df['Time'] = np.arange(len(df.index))

print(df['Time'].values)

# [0, 1, 2, 3, 4 ... , len(df.index) - 1]
```

### Lag feature


```python

df['Lag_1'] = df['target'].shift(1)

```
