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
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
# # Generate Training and Testing Data

# %%
feat_names = df.columns[:-1]
class_names = df['type'].unique()

# %%
X_train, X_test, y_train, y_test = train_test_split(df[feat_names],
                                                    df['type'],
                                                    test_size=0.25,
                                                    random_state=42)

# %% [markdown]
# # Training and Fitting the Model

# %%
k_neighbors_classifier = KNeighborsClassifier()
k_neighbors_classifier = k_neighbors_classifier.fit(X_train, y_train)

# %% [markdown]
# # Making Predictions

# %%
predictions = k_neighbors_classifier.predict(X_test)
probabilities = k_neighbors_classifier.predict_proba(X_test)
results_df = pd.DataFrame({'prob of no':probabilities[:,0],
                         'prob of yes':probabilities[:,1],
                         'predicted class':predictions,
                         'actual class': y_test})
results_df.head()

# %% [markdown]
# # Checking Accuracy

# %%
accuracy = k_neighbors_classifier.score(X_test, y_test)
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
DecisionBoundaryDisplay.from_estimator(k_neighbors_classifier, X_test, alpha=0.4, response_method="predict", ax=ax)
ax.scatter(yesses['x'], yesses['y'], c=yesses['color'], s=20, edgecolor="k", label='yes')
ax.scatter(nos['x'], nos['y'], c=nos['color'], s=20, edgecolor="k", label='no')
ax.legend(title='binary variable')
ax.set_title(f'K Nearest Neighbor Classifier $(Accuracy = {accuracy:.2f})$')
plt.show()

# %% [markdown]
# # Confusion Matrix

# %%
cm = confusion_matrix(y_test, predictions,
                      labels=class_names,
                      normalize='all')
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot(cmap='Greens')
plt.show()

# %%
