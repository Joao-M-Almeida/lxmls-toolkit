import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        for i_class in range(n_classes):
            prior[i_class] = sum(y == classes[i_class])/float(n_docs)

        print "Priors trained"

        likelihood_numerator = np.zeros((n_words, n_classes))

        for d in xrange(n_docs):
            likelihood_numerator[:, y[d]] += x[d,:][:, np.newaxis]
            #for w in xrange(n_words):
            #    likelihood_numerator[w, y[d]] += x[d, w]

        likelihood_denominator = np.sum(likelihood_numerator, axis=0)

        for i_class in xrange(n_classes):
            likelihood[:, i_class] = (likelihood_numerator[:, i_class] + 1) / float(likelihood_denominator[i_class] + n_words)

        print "likelihood trained"


        params = np.zeros((n_words+1, n_classes))
        for i in xrange(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
