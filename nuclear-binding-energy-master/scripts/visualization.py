# -*- coding: utf-8 -*-
# Script: visualization.py
# Generated automatically from BindingEnergy.ipynb

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Learning Curve (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.title("Learning Curve (MAE)")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"), dpi=300)
plt.show()

# Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Test)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_pred.png"), dpi=300)
plt.show()

# Residuals
residuals = y_test.values.reshape(-1, 1) - y_pred
plt.figure(figsize=(10, 5))
sns.histplot(residuals.flatten(), kde=True)
plt.title("Residual Distribution (Test)")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "residuals_hist.png"), dpi=300)
plt.show()

