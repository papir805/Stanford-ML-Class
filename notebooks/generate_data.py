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

# %%
import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

# %%
random_seed=1000
sample_size=250

# %% [markdown]
# # data1

# %%
mean = np.array([0,0])
cov = np.array([[1, 0], [0, 1]])
distr = multivariate_normal(cov=cov, mean=mean, seed=random_seed)
data1 = distr.rvs(size=sample_size)

# %%
plt.plot(data1[:,0], data1[:,1], 'o', markeredgewidth=0.5, markeredgecolor='black');

# %% [markdown]
# # data2

# %%
mean = np.array([2,-2])
cov = np.array([[2, 0], [0, 2]])
distr = multivariate_normal(cov=cov, mean=mean, seed=random_seed)
data2 = distr.rvs(size=sample_size)

# %%
plt.plot(data2[:,0], data2[:,1], 'o', markeredgewidth=0.5, markeredgecolor='black');

# %% [markdown]
# # Both datasets

# %%
fig, ax = plt.subplots(1,1)
ax.plot(data1[:,0], data1[:,1], 'o', markeredgewidth=0.5, markeredgecolor='black', alpha=0.5)
ax.plot(data2[:,0], data2[:,1], 'o', markeredgewidth=0.5, markeredgecolor='black', alpha=0.5)
plt.savefig('../images/scatter_with_colors.png', bbox_inches='tight');

# %% [markdown]
# # joining data

# %%
joined_x = np.append(data1[:,0], data2[:,0])
joined_y = np.append(data1[:,1], data2[:,1])
types = np.array(['yes'] * sample_size + ['no'] * sample_size)

# %%
plt.plot(joined_x, joined_y, 'o', markeredgewidth=0.5, markeredgecolor='black', alpha=0.5)
plt.savefig("../images/scatter.png", bbox_inches='tight');

# %% [markdown]
# # exporting to csv

# %%
df = pd.DataFrame({'x':joined_x, 'y':joined_y, 'type':types})
df.to_csv("../data/toy_data.csv")

# %%
