# Financial Anomaly Detection su AAPL (Simulato)

Rilevamento anomalie nei prezzi azionari di un titolo simulato ispirato ad Apple Inc.  
L'analisi combina metodi statistici e modelli non supervisionati per individuare outlier su serie temporali.

## Obiettivi

- Analizzare andamenti di prezzo in una logica daily trading.
- Applicare modelli di outlier detection per supportare decisioni di rischio.
- Visualizzare visivamente segnali anomali e pattern di rottura.

## Tecnologie

- Python 3
- Pandas, Numpy
- Scikit-learn (Isolation Forest)
- Matplotlib

## Struttura

📦 aapl-anomaly-detection
├── anomaly_detection_aapl.py
├── aapl_anomaly_detection_2023.csv
├── aapl_anomalies_2023.png
└── README.md


## Esecuzione

```bash
pip install pandas scikit-learn matplotlib
python anomaly_detection_aapl.py

## *OUTPUT*
aapl_anomalies_2023.png: evidenzia anomalie con colori distintivi.
aapl_anomaly_detection_2023.csv: include prezzi e flag di anomalia.
