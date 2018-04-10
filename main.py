from util import identifyMisLabeled
from dataset import generate_mislabeled_data
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

# Training Dataset created with some contamination level
X_orig, Y_orig = generate_mislabeled_data(100000, 0.4)

# Test Dataset created with no contamination level
X_test, Y_test = generate_mislabeled_data(10000, 0.0)

kf = KFold(n_splits=10)

# Array of mismatches is created to store the number of times
# element at that index is identified as mislabeled by different models.
mismatches = np.zeros(shape = Y_orig.shape)

# For each combination of splits using cross-validation
for train_index, test_index in kf.split(X_orig):
    X_train, X_test = X_orig[train_index], X_orig[test_index]
    Y_train, Y_test = Y_orig[train_index], Y_orig[test_index]

    # LogisticRegressionClassifier
    identifyMisLabeled(X_train, Y_train, X_test, Y_test, LogisticRegression(), test_index, mismatches)

    # RandomForestClassifier
    identifyMisLabeled(X_train, Y_train, X_test, Y_test, RandomForestClassifier(max_depth=2, random_state=0),
                       test_index, mismatches)

print('Original dataset size : ', X_orig.shape[0])

# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)
clf = LogisticRegression()
clf.fit(X_orig, Y_orig)
print('Score without removing Mislabeled Data: ', clf.score(X_test, Y_test))

# Removing all the entries which are counted as mislabed by both classifiers
indexes = np.where(mismatches > 1)[0]
X_orig = np.delete(X_orig, indexes, 0)
Y_orig = np.delete(Y_orig, indexes)

print('Dataset size after removing Mislabeled Data : ', X_orig.shape[0])

X_train,X_test,Y_train,Y_test = train_test_split(X_orig,Y_orig,test_size=0.33)
clf = LogisticRegression()
clf.fit(X_train, Y_train)
print('Score after removing Mislabeled Data: ', clf.score(X_test, Y_test))