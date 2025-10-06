# -*- coding: utf-8 -*-
# Script: evaluation.py
# Generated automatically from BindingEnergy.ipynb

eval_results = best_model.evaluate(X_test_proc, y_test_scaled, verbose=0)
print("\n========== Test Metrics (scaled target) ==========")
print(f"Loss (MSE): {eval_results[0]:.6f}")
print(f"MAE:        {eval_results[1]:.6f}")
print(f"RMSE:       {eval_results[2]:.6f}")

# Predictions -> inverse scale to original units
y_pred_scaled = best_model.predict(X_test_proc, verbose=0)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test.values.reshape(-1, 1) - y_pred))

print("\n========== Test Metrics (original target) ==========")
print(f"MAE:  {mae:.6f}")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R^2:  {r2:.6f}")

