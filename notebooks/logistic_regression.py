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
<<<<<<< HEAD
df['color'] = df[target_label].map({'no':'green', 'yes':'red'})

# %%
# colors = []
# for y in y_test:
#     if y == 'yes':
#         colors.append('green')
#     else:
#         colors.append('red')
# for_scatter = X_test.copy(deep=True)
        
# for_scatter[target_label] = y_test
# for_scatter['color'] = colors

yesses = df[df[target_label]=='yes']
nos = df[df[target_label]=='no']

fig, ax = plt.subplots(1,1)

ax.scatter(yesses[x_label], yesses[y_label], c=yesses['color'], s=20, edgecolor="k", label='yes', alpha=0.3)
ax.set_xlabel(x_label)

ax.scatter(nos[x_label], nos[y_label], c=nos['color'], s=20, edgecolor="k", label='no', alpha=0.6)

ax.legend(title='Engine Failure ')

ax.set_title('Failure $')

plt.show()

# %%
fig, ax = plt.subplots(1,1)


df.plot.scatter(x=x_label, 
                y=y_label, 
                ax=ax, 
                c=df[target_label].map({'no':'green', 'yes':'red'}),
                label=['no', 'yes'])
ax.legend();
=======
>>>>>>> parent of 8d10c0a... Changed to pressure vs temp

# %% [markdown]
# # Generate Training and Testing Data

# %%
# Separate feature names from class names
<<<<<<< HEAD
feat_names = [x_label, y_label]
unique_labels = df[target_label].unique()

# %%
X_train, X_test, y_train, y_test = train_test_split(df[feat_names],
                                                    df[target_label],
=======
feat_names = df.columns[:-1]
class_names = df['type'].unique()

# %%
X_train, X_test, y_train, y_test = train_test_split(df[feat_names],
                                                    df['type'],
>>>>>>> parent of 8d10c0a... Changed to pressure vs temp
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
        colors.append('green')
    else:
        colors.append('red')
for_scatter = X_test.copy(deep=True)
        
for_scatter['type'] = y_test
for_scatter['color'] = colors

yesses = for_scatter[for_scatter['type']=='yes']
nos = for_scatter[for_scatter['type']=='no']

# %%
fig, ax = plt.subplots(1,1)
DecisionBoundaryDisplay.from_estimator(log_reg_classifier, X_test, alpha=0.4, response_method="predict", ax=ax)
<<<<<<< HEAD
ax.scatter(yesses[x_label], yesses[y_label], c=yesses['color'], s=20, edgecolor="k", label='yes')
ax.scatter(nos[x_label], nos[y_label], c=nos['color'], s=20, edgecolor="k", label='no')
ax.legend(title='Failure?')
=======
ax.scatter(yesses['x'], yesses['y'], c=yesses['color'], s=20, edgecolor="k", label='yes')
ax.scatter(nos['x'], nos['y'], c=nos['color'], s=20, edgecolor="k", label='no')
ax.legend(title='binary variable')
>>>>>>> parent of 8d10c0a... Changed to pressure vs temp
ax.set_title(f'Logistic Regression Classifier $(Accuracy = {accuracy:.2f})$')
plt.tight_layout()
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
