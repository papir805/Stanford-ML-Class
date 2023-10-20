# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# %% [markdown]
# # Read Data

# %%
df = pd.read_csv("../data/toy_data.csv", index_col=0)

# %%
df.head()

# %%
x_label = df.columns[0]
y_label = df.columns[1]

sample_size = df.shape[0]

# %% [markdown]
# # Generate Training and Testing Data

# %%
X_train, X_test, y_train, y_test = train_test_split(df[x_label], df[y_label],
                                                    test_size=0.25,
                                                    random_state=42)

# %% [markdown]
# # Train Linear Regression Model

# %%
lin_reg_model = scipy.stats.linregress(X_train, y_train)

slope = lin_reg_model.slope
intercept = lin_reg_model.intercept
r_value = lin_reg_model.rvalue

# %% [markdown]
# # Make Predictions and Checking Error 

# %%
predictions = slope * X_test + intercept
residuals = y_test - predictions

# %%
results_df = pd.DataFrame({'x':X_test, 
                           'y_actual':y_test, 
                           'y_predicted':predictions,
                           'residual':residuals})
results_df.head()

# %% [markdown]
# # Root Mean Squared Error

# %%
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# %% [markdown]
# # Visualizing the Linear Model

# %%
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,6))

ax[0].scatter(X_test, y_test, label='Actual')
ax[0].set_xlabel(x_label)
ax[0].set_ylabel(y_label)
ax[0].set_title(f'Actual vs. Predicted Values ($n={sample_size})$')
ax[0].plot(X_test, predictions, color='r', label='predicted')
ax[0].legend()

ax[1].scatter(X_test, residuals, label='Actual')
ax[1].axhline(0, color='r', linestyle='--')
ax[1].set_xlabel(x_label)
ax[1].set_ylabel(f'Error in {y_label}')
ax[1].set_title(f'Residual Plot ($r={r_value:.4f}; RMSE={rmse:.2f})$')

plt.tight_layout()

# %% [markdown]
# # Using statsmodels

# %%
from statsmodels.formula.api import ols

# %%
