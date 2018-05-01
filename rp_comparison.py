from __future__ import print_function
from rankpruning.rankpruning import RankPruning
import rankpruning.other_pnlearning_methods as other_pnlearning_methods

import numpy as np

# Libraries uses only for the purpose of the tutorial
from numpy.random import multivariate_normal
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as acc
import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[3]:

# Create our training dataset having examples drawn from two 2-dimensional Guassian distributions.
# A Pandas DataFrame is used only for the purposes of demonstration. Numpy arrays are preferred.
# In this example, we allow for class imbalance (twice as many negative examples).
neg = pd.DataFrame(multivariate_normal(mean=[2,2], cov=[[10,-1.5],[-1.5,5]], size=1000), columns = ['x1', 'x2'])
neg['label'] = [0 for i in range(len(neg))]
pos = pd.DataFrame(multivariate_normal(mean=[5,5], cov=[[1.5,0.3],[1.3,4]], size=500), columns = ['x1', 'x2'])
pos['label'] = [1 for i in range(len(pos))]

try:
  # Plot the distribution for your viewing.
  get_ipython().magic(u'matplotlib inline')
  from matplotlib import pyplot as plt
  plt.figure(figsize=(7, 7))
  plt.scatter(pos['x1'], pos['x2'], c='blue', s=50, marker="+", linewidth=1)
  plt.scatter(neg['x1'], neg['x2'], s=50, facecolors='none', edgecolors='red', linewidth=1)
except:
  print("Plotting is only supported in an iPython interface.")


# Choose mislabeling noise rates.
frac_pos2neg = 0.4 # rh1, P(s=0|y=1) in literature
frac_neg2pos = 0.4 # rh0, P(s=1|y=0) in literature

# Combine data into training examples and labels
data = neg.append(pos)
X = data[["x1","x2"]].values
y = data["label"].values

# Noisy P̃Ñ learning: instead of target y, we have s containing mislabeled examples.
# First, we flip positives, then negatives, then combine.
# We assume labels are flipped by some noise process uniformly randomly within each class.
s = y * (np.cumsum(y) <= (1 - frac_pos2neg) * sum(y))
s_only_neg_mislabeled = 1 - (1 - y) * (np.cumsum(1 - y) <= (1 - frac_neg2pos) * sum(1 - y))
s[y==0] = s_only_neg_mislabeled[y==0]


# Create testing dataset
neg_test = multivariate_normal(mean=[2,2], cov=[[10,-1.5],[-1.5,5]], size=2000)
pos_test = multivariate_normal(mean=[5,5], cov=[[1.5,1.3],[1.3,4]], size=1000)
X_test = np.concatenate((neg_test, pos_test))
y_test = np.concatenate((np.zeros(len(neg_test)), np.ones(len(pos_test))))

# ## Comparing models using a logistic regression classifier.
# For shorter notation use rh1 and rh0 for noise rates.

clf = LogisticRegression()
rh1 = frac_pos2neg
rh0 = frac_neg2pos

models = {
  "Baseline" : other_pnlearning_methods.BaselineNoisyPN(clf),
  "Rank Pruning" : RankPruning(clf = clf),
  "Rank Pruning (noise rates given)": RankPruning(rh1, rh0, clf)
}

for key in models.keys():
  model = models[key]
  model.fit(X, s)
  pred = model.predict(X_test)
  pred_proba = model.predict_proba(X_test) # Produces P(y=1|x)

  print("\n%s Model Performance:\n==============================\n" % key)
  print("Accuracy:", acc(y_test, pred))

