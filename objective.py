'''
Implemented : Mamadou S.
Date : 2026, April 03
Definition: This is the central file that defines the mathematical problem to be solved.
Role: It contains the Objective class, which specifies the loss function 
      (here, the Hinge Loss for the SVM) and the penalty (L2). It also defines 
      how to evaluate the solver's performance using the evaluate_result method (calculating the primal objective).
'''

import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "SVM Binary Classification (no intercept)"

    parameters = {
        'C': [1., 0.1],
    }

    def __init__(self, C=1.):
        self.C = C

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_features = self.X.shape[1]

    def get_one_result(self):
        return dict(beta=np.zeros(self.n_features))

    def evaluate_result(self, beta):
        # We calculate the score for y * (X @ beta)
        projected_labels = self.X @ beta
        hinge_loss = np.maximum(1 - self.y * projected_labels, 0.).sum()
        
        loss = self.C * hinge_loss
        pen = 0.5 * np.dot(beta, beta)
        return loss + pen
    # def compute(self, beta):
    #    loss = self.C * np.sum(
    #        np.maximum(1 - self.y * (self.X @ beta), 0.)
    #    )
    #    pen = 0.5 * np.dot(beta, beta)
    #    return loss + pen

    def get_objective(self):
        return dict(X=self.X, y=self.y, C=self.C)
