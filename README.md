# Ethiopia Housing Dashboard

A Streamlit app that predicts real estate valuation in Ethiopian Birr using a multivariate linear regression model.

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app loads the trained model (`property_model.pkl`) and scaler (`property_scaler.pkl`); adjust the property specs in the sidebar to get live price predictions in ETB.

## Project Files

- `House-Price.ipynb` — model training notebook
- `app.py` — Streamlit dashboard
- `requirements.txt` — Python dependencies
- `houses_improved_data.csv` - dataset
