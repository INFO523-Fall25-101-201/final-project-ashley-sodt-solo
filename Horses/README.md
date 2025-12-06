# Horses (Raw Data)

This folder contains the raw EquiPro stride-level CSV result files for all
exercise sessions. These files come directly from the EquiPro system and must
be treated as read-only.

The raw data is organized by horse:
Horses/
├── Duque/
├── Jackson/
└── Perseo/


Each subfolder contains multiple session files named with a timestamp, for
example:
20240626T082743-Duque-results.csv
20241018T180653-Perseo-results.csv


These raw data files are:

- not edited  
- not cleaned manually  
- loaded directly by `analysis/01_features.qmd`  

All preprocessing and feature engineering steps are performed programmatically.

---

## Notes

- Do not add processed or modified data to this folder.
- Any updates to session-level data should be done by regenerating
  `data/session_features.csv`.

