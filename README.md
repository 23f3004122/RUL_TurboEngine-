Developed an advanced Remaining Useful Life (RUL) prediction model on NASA’s turbofan engine degradation dataset using LSTM.
Performed full preprocessing pipeline including normalization, interpolation for short sequences, and handcrafted feature fusion using
regression coefficients.
Integrated a custom 3D attention mechanism around LSTM layers with engineered sensor-level statistics to improve temporal feature
extraction.
Trained the model with dropout, early stopping, and custom callbacks; achieved RMSE as low as 25.9 across multiple runs.
Exported predicted results as .mat and .csv for cross-platform evaluation; visualized test loss, RMSE trend, and prediction curves over test
units.
