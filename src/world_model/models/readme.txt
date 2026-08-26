This directory stores world-model artifacts produced by:

  python -m src.world_model.train --npz data/processed/state_windows.npz

  world_lstm.pt   LSTM weights (next-state + attack-within-K heads)
  scaler.npz      train-only mean/std for the 32-d state vector
  metrics.json    test-set LSTM vs logistic regression comparison
