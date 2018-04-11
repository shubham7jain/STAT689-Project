from util import identifyMisLabeled
from dataset import generate_mislabeled_data, read_mnist_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

# Training Dataset created with some contamination level
# X_orig, Y_orig = generate_mislabeled_data(1000, 0.4)
#
# # Test Dataset created with no contamination level
# X_orig_test, Y_orig_test = generate_mislabeled_data(100, 0.0)

classifiers_for_filtering = [
    'LogisticRegression',
    'RandomForestClassifier',
    'MLPClassifier'
]

learning_algorithm = LogisticRegression(solver = 'lbfgs')
X_orig, X_orig_test, Y_orig, Y_orig_test = read_mnist_data(0.4)

kf = KFold(n_splits=5)

# Array of mismatches is created to store the number of times
# element at that index is identified as mislabeled by different models.
mismatches = np.zeros(shape = Y_orig.shape)

c = 1
# For each combination of splits using cross-validation
for train_index, test_index in kf.split(X_orig):
    print('Running Cross-Validation Split ', c)
    c += 1
    X_train, X_test = X_orig[train_index], X_orig[test_index]
    Y_train, Y_test = Y_orig[train_index], Y_orig[test_index]

    for classifier in classifiers_for_filtering:
        identifyMisLabeled(X_train, Y_train, X_test, Y_test, classifier, test_index, mismatches)

print('Original dataset size : ', X_orig.shape[0])

clf = LogisticRegression(solver = 'lbfgs')
clf.fit(X_orig, Y_orig)
print('Score without removing Mislabeled Data: ', clf.score(X_orig_test, Y_orig_test))

# Removing all the entries which are counted as mislabed by both classifiers
indexes = np.where(mismatches >= len(classifiers_for_filtering))[0]
X_new = np.delete(X_orig, indexes, 0)
Y_new = np.delete(Y_orig, indexes)

print('Dataset size after removing Mislabeled Data : ', X_new.shape[0])

clf = learning_algorithm
clf.fit(X_new, Y_new)
print('Score after removing Mislabeled Data: ', clf.score(X_orig_test, Y_orig_test))