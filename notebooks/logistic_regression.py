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
# # Introduction

# %%

# %% [markdown]
# # Imports 

# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# %% [markdown]
# # Read Data

# %%
df = pd.read_csv("../data/toy_data.csv", index_col=0)

# %%
df.head()

# %% [markdown]
# # Visualize The dataset

# %%
x_label = 'pressure (psi)'
y_label = 'temperature (deg F)'
target_label = 'failure'

# %%
df['color'] = df[target_label].map({'no':'green', 'yes':'red'})
df_yesses = df[df[target_label]=='yes']
df_nos = df[df[target_label]=='no']

# %%
fig, ax = plt.subplots(1,1)

ax.scatter(df_yesses[x_label], df_yesses[y_label], c=df_yesses['color'], s=20, edgecolor="k", label='yes', alpha=0.3)


ax.scatter(df_nos[x_label], df_nos[y_label], c=df_nos['color'], s=20, edgecolor="k", label='no', alpha=0.6)

ax.set_title('Engine Failure: Pressure vs. Temperature')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.legend(title='Failure?')

plt.show()

# %% [markdown]
# # Generate Training and Testing Data

# %%
# Separate feature names from class names
feat_names = [x_label, y_label]
unique_labels = df[target_label].unique()

# %%
X_train, X_test, y_train, y_test = train_test_split(df[feat_names],
                                                    df[target_label],
                                                    test_size=0.25,
                                                    random_state=42)

# %% [markdown]
# # Training and Fitting the Model

# %%
log_reg_classifier = LogisticRegression(random_state=0)
log_reg_classifier = log_reg_classifier.fit(X_train, y_train)

# %% [markdown]
# # Making Predictions

# %%
predictions = log_reg_classifier.predict(X_test)
probabilities = log_reg_classifier.predict_proba(X_test)
results_df = pd.DataFrame({'prob of no':probabilities[:,0],
                         'prob of yes':probabilities[:,1],
                         'predicted class':predictions,
                         'actual class': y_test})
results_df.head()

# %% [markdown]
# # Checking Accuracy

# %%
accuracy = log_reg_classifier.score(X_test, y_test)
accuracy

# %% [markdown]
# # Visualizing Decision Boundaries

# %%
colors = []
for y in y_test:
    if y == 'yes':
        colors.append('red')
    else:
        colors.append('green')
X_test_copy = X_test.copy(deep=True)
        
X_test_copy['type'] = y_test
X_test_copy['color'] = colors

test_yesses = X_test_copy[X_test_copy['type']=='yes']
test_nos = X_test_copy[X_test_copy['type']=='no']

# %%
fig, ax = plt.subplots(1,1)
DecisionBoundaryDisplay.from_estimator(log_reg_classifier, X_test, alpha=0.4, response_method="predict", ax=ax)

ax.scatter(test_nos[x_label], test_nos[y_label], c=test_nos['color'], s=20, edgecolor="k", label='yes')

ax.scatter(test_yesses[x_label], test_yesses[y_label], c=test_yesses['color'], s=20, edgecolor="k", label='no')

ax.legend(title='Failure?')
ax.set_title(f'Logistic Regression Classifier $(Accuracy = {accuracy:.2f})$')

#plt.tight_layout()
plt.show()

# %% [markdown]
# # Confusion Matrix

# %%
cm = confusion_matrix(y_test, predictions,
                      labels=unique_labels,
                      normalize='all')
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
cm_display.plot(cmap='Greens')
plt.show()

# %%
