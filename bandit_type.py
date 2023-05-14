import numpy as np
import distribution

class Linear():
    
    def __init__(self, X, Y, theta_star, T, sigma=1, name=""):
        
        self.X = X
        self.Y = Y
        self.n, self.d = X.shape
        self.theta_star = theta_star
        
        self.V = np.eye(self.d)

        self.T = T  # time steps
        self.sigma = sigma
        self.name = name
        self.arms_recommended = []


class Concept():
    def __init__(self, X, gen_star, pi, T, sigma=1, name=""):    
        self.X = X
        self.n = X.shape[0]
        if type(gen_star) is not distribution.GenericFunction:
            raise Exception('gen_star must be a GenericFunction object')
        self.gen_star = gen_star
        self.pi = pi
        
        self.T = T  # time steps
        self.sigma = sigma
        self.name = name
        self.arms_recommended = []