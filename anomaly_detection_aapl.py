# anomaly_detection_aapl.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# -----------------------------
# Progetto: Financial Market Anomaly Detection
# Dati mock AAPL 2023 + Z-score + Isolation Forest
# -----------------------------

# Step 1: Simulazione dati storici AAPL per il 2023
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
prices = np.cumsum(np.random.normal(loc=0.1, scale=2.0, size=len(dates))) + 150

# Creazione DataFrame
stock_df = pd.DataFrame({'Date': dates, 'Close': prices})
stock_df.set_index('Date', inplace=True)

# Step 2: Bollinger Bands
rolling_window = 20
stock_df['RollingMean'] = stock_df['Close'].rolling(
    window=rolling_window).mean()
stock_df['RollingStd'] = stock_df['Close'].rolling(window=rolling_window).std()
stock_df['UpperBand'] = stock_df['RollingMean'] + 2 * stock_df['RollingStd']
stock_df['LowerBand'] = stock_df['RollingMean'] - 2 * stock_df['RollingStd']

# Step 3: Z-score Anomaly Detection
stock_df['Z-Score'] = (stock_df['Close'] -
                       stock_df['RollingMean']) / stock_df['RollingStd']
stock_df['Z_Anomaly'] = stock_df['Z-Score'].abs() > 2.5

# Step 4: Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
stock_df['IF_Anomaly'] = model.fit_predict(stock_df[['Close']]) == -1

# Step 5: Visualizzazione
plt.figure(figsize=(14, 7))
plt.plot(stock_df.index, stock_df['Close'],
         label='Close Price', color='blue', alpha=0.6)
plt.scatter(stock_df.index[stock_df['Z_Anomaly']], stock_df['Close'][stock_df['Z_Anomaly']],
            color='orange', label='Z-score Anomalies', marker='o')
plt.scatter(stock_df.index[stock_df['IF_Anomaly']], stock_df['Close'][stock_df['IF_Anomaly']],
            color='red', label='Isolation Forest Anomalies', marker='x')
plt.fill_between(stock_df.index, stock_df['UpperBand'], stock_df['LowerBand'],
                 color='lightgray', alpha=0.2, label='Bollinger Bands')
plt.title("📈 AAPL Anomaly Detection (Mock 2023)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Salvataggio
plt.savefig("aapl_anomalies_2023.png")
stock_df.to_csv("aapl_anomaly_detection_2023.csv")
