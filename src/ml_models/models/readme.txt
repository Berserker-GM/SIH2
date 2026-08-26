This directory stores binary model artifacts that are NOT committed to git.

  lstm_ae_weights.pt  — saved by train/train_lstm.py
  iforest.pkl         — saved by train/train_iforest.py

To generate them:

  cd normative/ml_models
  python train/train_lstm.py   --csv /path/to/cicids2018.csv --epochs 30
  python train/train_iforest.py --csv /path/to/cicids2018.csv

Both scripts accept CICIDS2018 or UNSW-NB15 CSVs.
Make sure the CSV has a "Label" column with "BENIGN" values.
