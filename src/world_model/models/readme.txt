This directory stores world-model artifacts produced by:

  python -m src.world_model.train --npz data/processed/state_windows.npz

v1 (CIC-only, quoted 01 Mar numbers — do not overwrite):
  world_lstm.pt   LSTM weights
  scaler.npz      train-only mean/std
  metrics.json    test-set LSTM vs logistic regression

v2 (CIC + persona, --tag v2):
  world_lstm_v2.pt
  scaler_v2.npz
  metrics_v2.json
  stage_decoder_v2.joblib

