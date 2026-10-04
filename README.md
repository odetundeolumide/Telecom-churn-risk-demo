# Telecom Churn Risk Demo

Streamlit app + training script. **Demo only, not for real retention decisions.**

## Run locally
    pip install -r requirements.txt
    python train.py telecom_churn.csv     # creates model.joblib + metrics.json (already included)
    streamlit run app.py

## Deploy free (Streamlit Community Cloud)
1. Push this folder to a GitHub repo (keep `model.joblib` and `metrics.json`; do not commit the CSV).
2. On share.streamlit.io: New app, pick the repo, main file `app.py`, Deploy.

## What train.py does differently from the notebook
- Stratified 80/20 split (the notebook's was not stratified).
- All metrics use (y_true, y_pred) order. The notebook's tuned-model report had them swapped,
  which made recall look like 0.84 when it was about 0.62.
- Quotes a repeated cross-validation AUC (about 0.90) instead of one split.
- Alert threshold chosen on out-of-fold training predictions, never on the test set.

## Data
TODO: add the dataset name and link here before publishing.
