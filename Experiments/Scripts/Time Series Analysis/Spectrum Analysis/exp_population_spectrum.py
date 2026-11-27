"""
Phase 2: Modern Pipeline Integration (Python)
Experiment: Population Spectrum & Time Series Forecasting
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from darts import TimeSeries
    from darts.models import LightGBMModel
    HAS_DARTS = True
except ImportError:
    HAS_DARTS = False
    print("Darts not found. Falling back to simple Pandas rolling models for demonstration.")

print("=== Phase 2: Python MLOps & Advanced Forecasting ===")

# Step 2.1: Data Ingestion
print("2.1 Loading Parquet datasets...")
train_path = 'lake/processed/train.parquet'
test_path = 'lake/processed/test.parquet'

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Missing {train_path}. Run MATLAB script first.")

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Set DatetimeIndex
df_train.set_index('timestamps', inplace=True)
df_test.set_index('timestamps', inplace=True)
df_train.index.freq = 'D'
df_test.index.freq = 'D'

print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

# Step 2.1.5: Exploratory Data Analysis & Verification
print("2.1.5 Verifying Population Spectrum (Power Spectral Density)...")
sunspots_centered = df_train['SunspotNumber'].values - df_train['SunspotNumber'].mean()
fs = 1.0

# Exact variance equality check
f_exact, pxx_exact = signal.periodogram(sunspots_centered, fs, window='boxcar', scaling='density')
df_freq = f_exact[1] - f_exact[0]
area_psd = np.sum(pxx_exact) * df_freq
total_variance = np.var(sunspots_centered, ddof=0)

print("Verification:")
print(f"  Total Area under Spectrum: {area_psd:.4f}")
print(f"  Total Variance:            {total_variance:.4f}")
print(f"  Difference:                {abs(area_psd - total_variance):.4e}")

# Welch's estimate for plotting
f_welch, pxx_welch = signal.welch(sunspots_centered, fs, nperseg=4096)
period_years = (1 / f_welch[1:]) / 365.25 # Ignore DC to avoid divide by zero

# Step 2.2: Time Series Cross-Validation & Grid Search for ARMA(p, q)
print("2.2 Performing Time Series Cross-Validation & Grid Search for ARMA(p, q)...")
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Resample to monthly to match MATLAB's retime(..., 'monthly', 'mean')
df_train_monthly = df_train.resample('MS')['SunspotNumber'].mean().dropna()
df_test_monthly  = df_test.resample('MS')['SunspotNumber'].mean().dropna()

train_vals = df_train_monthly.values
max_p, max_q = 5, 5
tscv = TimeSeriesSplit(n_splits=5)

results = []

print("  Searching orders (p, q) in [0..5] x [0..5]...")
for p in range(max_p + 1):
    for q in range(max_q + 1):
        if p == 0 and q == 0:
            continue
        
        cv_maes = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 5-Fold Time Series Split Cross-Validation
            for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_vals)):
                y_tr, y_val = train_vals[tr_idx], train_vals[val_idx]
                try:
                    m = ARIMA(y_tr, order=(p, 0, q)).fit()
                    fc = m.forecast(steps=len(y_val))
                    cv_maes.append(mean_absolute_error(y_val, fc))
                except Exception:
                    pass
            
            # Fit on full training set to get AIC and BIC
            try:
                full_m = ARIMA(train_vals, order=(p, 0, q)).fit()
                aic = full_m.aic
                bic = full_m.bic
            except Exception:
                aic, bic = np.nan, np.nan
        
        if cv_maes:
            mean_cv_mae = np.mean(cv_maes)
            results.append({
                'p': p, 'q': q, 
                'cv_mae': mean_cv_mae, 
                'aic': aic, 'bic': bic
            })

# Sort candidate models by CV MAE
results_df = pd.DataFrame(results).sort_values('cv_mae')

print("\n--- Top 5 ARMA(p, q) Models by 5-Fold Cross-Validation MAE ---")
print(f" {'p':<3} {'q':<3} {'CV MAE':<10} {'AIC':<10} {'BIC':<10}")
print("-" * 40)
for _, row in results_df.head(5).iterrows():
    print(f" {int(row['p']):<3} {int(row['q']):<3} {row['cv_mae']:<10.2f} {row['aic']:<10.2f} {row['bic']:<10.2f}")

best_p = int(results_df.iloc[0]['p'])
best_q = int(results_df.iloc[0]['q'])
print(f"\nBest model selected by CV: ARMA({best_p}, 0, {best_q})")

# Fit best selected ARMA model on full training set
best_arma = ARIMA(df_train_monthly.values, order=(best_p, 0, best_q)).fit()
print(f"Estimated ARMA({best_p},{best_q}) model variance: {best_arma.params[-1]:.4f}")

n_fcast = len(df_test_monthly)
y_pred_monthly = best_arma.forecast(steps=n_fcast)
y_true_monthly = df_test_monthly.values
fcast_index = df_test_monthly.index
print(f"Forecasting {n_fcast} months using optimal ARMA({best_p}, {best_q})...")

# Step 2.4: Evaluation
mae = mean_absolute_error(y_true_monthly, y_pred_monthly)
rmse = np.sqrt(mean_squared_error(y_true_monthly, y_pred_monthly))

print("2.4 Evaluation Metrics (monthly):")
print(f"  MAE:  {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")

# Set OceanViz Dark Theme properties for matplotlib
plt.rcParams.update({
    'figure.facecolor': (0.11, 0.11, 0.13),
    'axes.facecolor': (0.11, 0.11, 0.13),
    'axes.edgecolor': (0.90, 0.90, 0.90),
    'axes.labelcolor': (0.90, 0.90, 0.90),
    'text.color': (0.90, 0.90, 0.90),
    'xtick.color': (0.90, 0.90, 0.90),
    'ytick.color': (0.90, 0.90, 0.90),
    'grid.color': (0.25, 0.25, 0.25),
    'grid.alpha': 0.5,
    'legend.facecolor': (0.11, 0.11, 0.13),
    'legend.edgecolor': (0.25, 0.25, 0.25),
    'lines.linewidth': 1.2
})

oceanviz_primary = (0.20, 0.60, 1.00)
oceanviz_secondary = (1.00, 0.40, 0.40)

# Optional: Plot
fig = plt.figure(figsize=(10, 12))

# Subplot 1: Daily Sunspots
ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(df_train.index, df_train['SunspotNumber'], color=oceanviz_primary)
ax1.set_title('Daily Sunspots (Train)')
ax1.set_ylabel('Sunspot Number')
ax1.grid(True)

# Subplot 2: Autocorrelation (20 Years)
max_lag = 365 * 20
# Compute ACF using correlate
acf = np.correlate(sunspots_centered, sunspots_centered, mode='full')
acf = acf[len(acf)//2:] / acf[len(acf)//2]
acf = acf[:max_lag]
lags_arr = np.arange(max_lag)

ax2 = fig.add_subplot(4, 1, 2)
# Using stem-like or just plot since it's dense
ax2.plot(lags_arr, acf, color=oceanviz_secondary, linewidth=1.0)
ax2.set_xlim(0, max_lag)
ax2.set_title('Autocorrelation (20 Years)')
ax2.set_xlabel('Lag (Days)')
ax2.set_ylabel('Sample Autocorrelation')
ax2.grid(True)

# Subplot 3: Population Spectrum
ax3 = fig.add_subplot(4, 1, 3)
ax3.plot(period_years, 10 * np.log10(pxx_welch[1:]), color=oceanviz_secondary, linewidth=1.5)
ax3.set_xlim(0.5, 20)
ax3.set_title("Welch's Power Spectral Density Estimate")
ax3.set_xlabel('Period (Years)')
ax3.set_ylabel('Power/Frequency (dB/day^{-1})')
ax3.grid(True)

# Subplot 4: Forecast (monthly, matching MATLAB ARMA)
ax4 = fig.add_subplot(4, 1, 4)
ax4.plot(fcast_index, y_true_monthly, label='Actual (Test - Monthly)', color=oceanviz_primary)
ax4.plot(fcast_index, y_pred_monthly, label=f'Predicted ARMA({best_p},{best_q})', color=oceanviz_secondary, linewidth=1.5)
ax4.set_title(f'Sunspot Number Forecast ({n_fcast} months) — ARMA({best_p},{best_q})')
ax4.set_xlabel('Date')
ax4.set_ylabel('Sunspot Number')
ax4.grid(True)
ax4.legend()

plt.tight_layout()
os.makedirs('lake/processed', exist_ok=True)
output_img = 'lake/processed/forecast_result.png'
plt.savefig(output_img, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
print(f"Plots saved to {output_img}")
plt.show() # Commented out to avoid blocking
print("Python Phase Complete.")
