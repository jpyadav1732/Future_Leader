# Identifying Future Leaders Through Data Analytics

Working Python implementation for generating employee data, running EDA, training ML models, selecting the best model, and predicting leadership potential.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run full workflow

```bash
python future_leader_pipeline.py
```

## Run dashboard

```bash
streamlit run streamlit_app.py
```

The pipeline creates:

- `data/future_leaders_dataset.csv`
- `plots/*.png`
- `models/future_leader_model.joblib`
- `models/model_metadata.joblib`
