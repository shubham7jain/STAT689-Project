from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def identifyMisLabeled(X_train, Y_train, X_test, Y_test, classifierName, test_index, mismatches):
    clf = None
    if classifierName == 'LogisticRegression':
        clf = LogisticRegression(solver = 'lbfgs')
    elif classifierName == 'RandomForestClassifier':
        clf = RandomForestClassifier(max_depth=2, random_state=0)
    elif classifierName == 'SVC':
        clf = SVC(kernel='linear')
    elif classifierName == 'GaussianNB':
        clf = GaussianNB()
    elif classifierName == 'MLPClassifier':
        clf = MLPClassifier(alpha=1)

    # train scikit learn model
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    i = 0
    for d, c in zip(Y_test, Y_pred):
        if(d != c):
            mismatches[test_index[i]] += 1
        i += 1
