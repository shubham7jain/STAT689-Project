from sklearn.linear_model import LogisticRegression
from rankpruning import assert_inputs_are_valid

## Uses logistic regression as default.
class BaselineNoisyPN:
  '''BaselineNoisyPN fits the classifier using noisy labels (assumes s = y).
  '''

  def __init__(self, clf = None):

    # Stores the classifier used.
    # Default classifier used is logistic regression
    self.clf = LogisticRegression() if clf is None else clf


  def fit(self, X, s):
    '''Train the classifier clf with s labels.

    X : np.array
      Input feature matrix (N, D), 2D numpy array
    s : np.array
      A binary vector of labels, s, which may contain mislabeling
    '''

    assert_inputs_are_valid(X, s)

    self.clf.fit(X, s)


  def predict(self, X):
    '''Returns a binary vector of predictions.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    '''

    return self.clf.predict(X)


  def predict_proba(self, X):
    '''Returns a vector of probabilties for only P(y=1) for each example in X.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    '''

    return self.clf.predict_proba(X)[:,1]

