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

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt

import graphviz

# %% [markdown]
# # Read data

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
decision_tree_classifier = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
decision_tree_classifier = decision_tree_classifier.fit(X_train, y_train)

# %% [markdown]
# # Visualizing the Decision Tree

# %%
fig, ax = plt.subplots(1,1)
tree.plot_tree(decision_tree_classifier, ax=ax);

# %% [markdown]
# # Visualizing Decision Tree using Graphviz

# %%
dot_data = tree.export_graphviz(decision_tree_classifier, out_file=None,
                                feature_names=feat_names,
                                class_names=class_names,
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("../images/test")

graph

# %% [markdown]
# # Making predictions

# %%
predictions = decision_tree_classifier.predict(X_test)
probabilities = decision_tree_classifier.predict_proba(X_test)
results_df = pd.DataFrame({'prob of no':probabilities[:,0],
                         'prob of yes':probabilities[:,1],
                         'predicted class':predictions,
                         'actual class': y_test})
results_df.head()

# %% [markdown]
# # Checking Accuracy

# %%
accuracy = decision_tree_classifier.score(X_test, y_test)
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
DecisionBoundaryDisplay.from_estimator(decision_tree_classifier, X_test, alpha=0.4, response_method="predict", ax=ax)
ax.scatter(yesses['x'], yesses['y'], c=yesses['color'], s=20, edgecolor="k", label='yes')
ax.scatter(nos['x'], nos['y'], c=nos['color'], s=20, edgecolor="k", label='no')
ax.legend(title='binary variable')
ax.set_title(f'Decision Tree Classifier $(Accuracy = {accuracy:.2f})$')
plt.show()

# %%
