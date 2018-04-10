import numpy as np

## computes the p_i for any x_i and beta
def logistic(x,beta):
    return( 1 / (1+np.exp(-np.sum(x*beta))))

def generate_mislabeled_data(n, alpha):
    x = np.linspace(-5,5,num=n)
    beta = np.array([3.,1.])
    X = np.column_stack((np.ones(n),x))

    p = np.apply_along_axis(lambda y: logistic(y,beta),1,X)

    ## simulate responses
    y = np.random.binomial(n=1,p=p)

    ## contamination level alpha
    z = np.random.binomial(n=1,p=1-alpha,size=n)
    r = np.random.binomial(n=1,p=0.5,size=n)

    ## we observe data w
    w = y*z + r*(1-z)

    return X, w