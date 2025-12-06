# Analysis

This folder contains the analysis notebooks used to process raw EquiPro data,
engineer features, model baselines, and perform anomaly detection.

All analysis is performed in **Quarto (`.qmd`) notebooks**, ensuring reproducibility.

---

## Files in this folder

### **`01_features.qmd`**
This notebook loads all raw EquiPro session CSV files from the `Horses/`
directory and computes session-level summary features, including:

- stride/stance/flight duration statistics  
- symmetry metrics  
- power metrics  
- interquartile ranges  
- left–right ratios  
- gait-phase proportions  

The notebook outputs a single processed dataset:
`data/session_features.csv`

### **`02_anomalies.qmd`**
This notebook performs the anomaly detection pipeline, including:

- per-horse baseline standardization  
- z-score thresholding  
- k-means clustering + distance-based anomaly scoring  
- Isolation Forest anomaly scoring  
- comparing agreement between methods  
- identifying and visualizing anomalous sessions  

### **`README.md`**
You are reading this file.

---

## Workflow

1. Run `01_features.qmd` to generate processed session-level features.  
2. Run `02_anomalies.qmd` to analyze deviations and detect anomalous sessions.

---

## Notes

- These notebooks should be run in order.
- All output figures should be saved in the `images/` directory.
- No raw data files are modified during analysis.