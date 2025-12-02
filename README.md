---

# Time Series Forecasting Project — JJ EPS & Amazon Stock

This project explores classical and deep learning approaches to time series forecasting using two real-world datasets.

- Johnson & Johnson (JJ) quarterly earnings per share (EPS)
- Amazon (AMZN) daily closing stock prices

Both case studies follow a similar pipeline

- Exploratory time series visualization
- Seasonal decomposition
- Stationarity testing (ADF) and differencing
- ACF/PACF analysis
- ARIMA model selection (grid search + auto-ARIMA)
- Final ARIMA model fit and forecasting
- LSTM-based forecasting as a deep learning alternative
- FFT-based frequency-domain analysis



---

## 📂 Project Structure

You might have the code organized roughly as:

* `part1_jj.ipynb` — JJ EPS analysis & forecasting
* `part2_amzn.ipynb` — Amazon stock analysis & forecasting
* `jj.csv` — Johnson & Johnson EPS data
* `AMZN.csv` — Amazon daily stock data
* `decomposejj.png` — JJ seasonal decomposition plot
* `sales_pred.png` — JJ ARIMA forecast with CI
* `lstmjj.png` — JJ LSTM 24-step forecast
* `lstmamz.png` — AMZN LSTM forecast (≈2 years)

(Names can be adjusted to match your actual files.)

---

## 🔧 Dependencies

Install required Python packages:

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn pmdarima tqdm tensorflow
```

**Main libraries used:**

* **Core:** `numpy`, `pandas`
* **Plots:** `matplotlib`
* **Time Series:** `statsmodels` (ARIMA, SARIMAX, ADF, seasonal_decompose)
* **Model selection:** `pmdarima` (`auto_arima`)
* **Metrics & scaling:** `scikit-learn` (`MinMaxScaler`, `mean_squared_error`)
* **Deep learning:** `tensorflow.keras` (LSTM, Dense, Sequential)
* **Progress bars:** `tqdm`
* **Frequency analysis:** `numpy.fft` (`fft`, `ifft`)

---

## 📊 Dataset 1 — Johnson & Johnson EPS (Quarterly)

### Data

* **File:** `jj.csv`
* **Columns:**

  * `date` — Quarterly dates
  * `data` — EPS values (Johnson & Johnson)

The series spans 84 quarterly observations.

### Workflow

#### 1️⃣ Visualization

* Load data, set `date` as DateTime index
* Plot EPS over time to observe trend and clear seasonality

#### 2️⃣ Seasonal Decomposition

* Multiplicative decomposition with quarterly period:

  ```python
  seasonal_decompose(data['data'], model='multiplicative', period=4)
  ```

* Plots:

  * Observed
  * Trend
  * Seasonal
  * Residuals

* Figure saved as **`decomposejj.png`**

#### 3️⃣ Stationarity & Transformations

* ADF test on original series → **non-stationary** (p ≈ 1.0)

* Apply log transform & first difference:

  ```python
  data['data_log'] = np.log(data['data'])
  data['data_tr_1'] = data['data_log'].diff()
  ```

* ADF on transformed series → **stationary** (p < 0.001), supporting **d = 1** for ARIMA.

#### 4️⃣ ACF & PACF

* Plots computed on the differenced log series to guide AR (p) and MA (q) orders.

#### 5️⃣ ARIMA Model Selection (Grid Search)

* Search over p, q ∈ {0,…,7} with d = 1

  ```python
  order_list = [(p,1,q) for p in range(8) for q in range(8)]
  ```

* For each order, fit ARIMA and record AIC.

* Best model (lowest AIC):

  ```text
  ARIMA(6,1,3) with AIC ≈ 115.50
  ```

#### 6️⃣ Auto-ARIMA Check

* `pmdarima.auto_arima` is run on a train subset and also supports (6,1,3) as optimal under constraints, reinforcing the grid search result.

#### 7️⃣ Final ARIMA(6,1,3) Model

* Fitted on full `data['data']`:

  ```python
  best_model = ARIMA(data['data'], order=(6,1,3))
  best_model_fit = best_model.fit()
  ```

* Residual diagnostics:

  * Ljung–Box test
  * Jarque–Bera test
  * Residual plots

These indicate no major autocorrelation left and decent residual behavior (though not perfectly normal).

#### 8️⃣ In-Sample Fit & Metrics

* Predictions over the entire historical range.
* Metrics used:

  * MAPE
  * MAE
  * MPE
  * RMSE
  * Correlation
  * MinMax error

Representative results:

* RMSE ≈ **0.401**
* MAPE ≈ **8.95%**
* Corr ≈ **0.996**

#### 9️⃣ Forecasting Future Quarters

* `get_forecast(steps=26)` for the next 26 quarters:

  * Predicted mean
  * 95% confidence intervals

* Forecast index built via quarterly DateTime, and plots include:

  * Historical EPS
  * Future predictions
  * Confidence interval band

* Figure saved as **`sales_pred.png`**

---

## 🤖 LSTM on JJ EPS

### Preprocessing

* Raw values reshaped to `(n,1)`
* `MinMaxScaler` used for normalization
* Sliding window: `window_size = 24`, `forecast_steps = 24` (24-step-ahead multi-output forecast).

### Model

* Stacked LSTM with Dense layers:

  ```python
  model = Sequential([
      LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
      LSTM(32),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(forecast_steps)
  ])
  ```

* Trained for 50 epochs with Adam optimizer.

### Performance & Plots

* Rolling 24-step forecast RMSE ≈ **1.636**

* Plots:

  * LSTM forecast vs actual for evaluation
  * Extended 24-month forecast appended to the original series

* Figure saved as **`lstmjj.png`**

---

## 🔉 FFT Analysis for JJ EPS

* FFT applied to the EPS series to inspect dominant frequencies:

  ```python
  X_jj = fft(jj_array)
  ```

* Plots:

  * FFT amplitude spectrum (stem plot)
  * IFFT reconstruction to confirm original signal recovery

This helps confirm the presence of strong seasonal components.

---

## 📊 Dataset 2 — Amazon (AMZN) Daily Close Prices

### Data

* **File:** `AMZN.csv`
* **Columns:**

  * `Date` — Trading date
  * `Close` — Daily closing stock price

The series contains ~1,258 daily data points.

### Workflow

#### 1️⃣ Visualization

* Load AMZN data, keep `Date` and `Close`.
* Plot the closing prices to inspect trend, volatility, and structural breaks.

#### 2️⃣ Seasonal Decomposition

* Multiplicative decomposition with `period=12`:

  ```python
  result_amazon = seasonal_decompose(data['Close'], model='multiplicative', period=12)
  ```

* Same 4 components plotted: observed, trend, seasonal, residuals.

#### 3️⃣ Stationarity & Differencing

* ADF test on `Close`:

  * Statistic ≈ -1.66
  * p-value ≈ 0.45 → non-stationary

* First difference:

  ```python
  data['Close_diff'] = data['Close'].diff()
  ```

* ADF on `Close_diff`:

  * Statistic ≈ -36.25
  * p-value = 0.0 → stationary

Thus **d = 1** is appropriate.

#### 4️⃣ ACF & PACF

* ACF and PACF computed on `Close_diff` to explore AR and MA orders.

#### 5️⃣ ARIMA Model Selection (Grid Search)

* Similar `optimize_ARIMA` function used as in the JJ example.
* Candidate orders: p, q ∈ {0,…,7} with d=1.
* Best model by AIC:

  ```text
  ARIMA(2,1,2) with AIC ≈ 6118.42
  ```

#### 6️⃣ Auto-ARIMA Baseline

* `auto_arima` on a train subset selects a simpler ARIMA(0,1,0) model, but the full grid search on complete data supports ARIMA(2,1,2) as superior.

#### 7️⃣ Final ARIMA(2,1,2) Model

* Fit:

  ```python
  best_model = ARIMA(data['Close'], order=(2,1,2))
  best_model_fit = best_model.fit()
  ```

* Summary includes AR/MA coefficients and diagnostics (Ljung–Box, JB, heteroskedasticity).

#### 8️⃣ In-Sample Fit & Metrics

* Predictions over the full historical index.
* Metrics like MAPE, MAE, RMSE, correlation are computed:

  * RMSE ≈ **0.3123**
  * MAPE ≈ **0.21%**
  * Corr ≈ **0.99996**

The in-sample fit is extremely tight.

#### 9️⃣ Long-Horizon ARIMA Forecast (504 Steps)

* `get_forecast(steps=504)` (~2 years of trading days).
* Combine predicted mean with lower/upper confidence intervals.
* Plot:

  * Historical AMZN close prices
  * Forecasted path
  * Shaded region for 95% CI

(You can specify a filename in `plt.savefig()` to store this plot.)

---

## 🤖 LSTM on AMZN

### Preprocessing

* Use only `Close` values.
* Normalize with `MinMaxScaler`.
* Build sequences with `time_steps = 60` (use 60 past days to predict next day).

```python
X, y = create_sequences(scaled_data, time_steps)
train_size = int(len(X) * 0.9)
X_train, y_train = X[:train_size], y[:train_size]
```

### Model

* Stacked LSTM:

  ```python
  model = Sequential([
      LSTM(100, return_sequences=True, activation='relu', input_shape=(time_steps, 1)),
      LSTM(50, activation='relu'),
      Dense(1)
  ])
  ```

* Trained for 50 epochs with batch size 32.

* Training loss steadily decreases to ~1e-3 range.

### Recursive Forecasting (≈2 Years)

* Start from the last 60 normalized observations.
* Predict 1-step ahead, append, slide window, repeat.
* `future_steps = 504` (≈ 2 years of business days).
* Inverse transform predictions back to price scale.
* Build forecast index with business-day frequency and combine with historical data.

Plot:

* Original AMZN `Close`
* LSTM forecast (dashed, dark orange)

Saved as: **`lstmamz.png`**

---

## 🔉 FFT Analysis for AMZN

* FFT applied to daily closing prices:

  ```python
  X = fft(close_array)
  ```

* Plots:

  * Amplitude spectrum (stem plot)
  * IFFT reconstruction to confirm signal consistency

This gives a frequency-domain perspective on the stock’s price behavior.

---

## ⚖️ ARIMA vs LSTM — High-Level Comparison

* **ARIMA**

  * Pros: Interpretable coefficients, confidence intervals, well-suited to linear structures and low-frequency series (like JJ EPS).
  * Cons: Limited in capturing complex nonlinear patterns; assumptions may break in highly volatile financial data.

* **LSTM**

  * Pros: Flexible for nonlinear dynamics and long-range dependencies; can handle complex patterns in stock prices.
  * Cons: Less interpretable, needs more data & tuning, no built-in statistical confidence intervals.

In both JJ and AMZN cases:

* ARIMA produces strong in-sample fits with low RMSE and high correlation.
* LSTM offers an alternative approach, especially for richer horizons and potentially capturing nonlinear behavior, at the cost of interpretability.

---

## 🚀 How to Run

1. Ensure `jj.csv` and `AMZN.csv` are in your working directory.

2. Open the notebooks/scripts for each part (e.g. `Assignment1.ipynb` and `Assignment1_part2.ipynb`).

3. Run all cells in order:

   * Data loading & visualization
   * Decomposition & stationarity tests
   * ARIMA model selection & forecast
   * LSTM training & forecast
   * FFT analysis

4. Check the generated plots:

   * `decomposejj.png`, `sales_pred.png`, `lstmjj.png`
   * `lstmamz.png` (and any ARIMA forecast plot you save for AMZN)

---

