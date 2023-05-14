import numpy as np
from catboost import CatBoostRegressor

def get_distribution(dist):
    if dist['name'] == 'Kitten':
        return Kitten(np.empty((0,dist['d'])), np.empty((0)), None)
    elif dist['name'] == 'Gaussian':
        d = dist['d']
        return Gaussian(np.zeros(d), np.eye(d))


class Distribution:
    pass


class GenericFunction():
    def __init__(self, f, sigma=1):
        self.f = f
        self.sigma = sigma

    def pull(self, x):
        """
        Evaluate the function f at x and add random noise.
        """
        noise = np.random.randn(*x.shape[:-1], 1) * self.sigma
        return self.evaluate(x) + noise.squeeze(axis=-1)

    def evaluate(self, x):
        """
        Evaluate the function f at x.
        """
        y = self.f(x)
        if type(y) is list or type(y) is np.ndarray:
            if len(y) == 1:
                y = y[0]
        return y



class Gaussian(Distribution):
    def __init__(self, theta, V, Vinv=None, S=0, sigma=1):
        super().__init__()
        self.theta = theta
        self.V = V
        self.S = S
        self.sigma = sigma
        if Vinv is None:
            self.Vinv = np.linalg.inv(V)
        else:
            self.Vinv = Vinv

    
    def update_posterior(self, x, y, copy=False):
        if copy:
            V = self.V + np.outer(x, x)
            S = self.S + np.dot(x, y)
            theta = np.linalg.inv(V) @ S
            return Gaussian(theta, V, Vinv=np.linalg.inv(V), S=S)
        else:
            self.V += np.outer(x, x)
            self.S += x * y
            self.Vinv = np.linalg.inv(self.V)
            self.theta = self.Vinv @ self.S
    
    
    def sample(self, k=1):
        if k==1:
            theta_tilde = np.random.multivariate_normal(self.theta, self.Vinv)
        else:
            theta_tilde = np.random.multivariate_normal(self.theta, self.Vinv, size=k)
        # def f(x):
        #     return x @ theta_tilde
        if k==1:
            return GenericFunction(lambda x: x@theta_tilde, self.sigma)

        return [GenericFunction(lambda x: x@theta.T, self.sigma) for theta in theta_tilde] 
    
    def map(self):
        return GenericFunction(lambda x: x@self.theta, self.sigma)
    


class Kitten(Distribution):
    def __init__(self, Xtrain, Ytrain, f, sigma=.5):
        super().__init__()
        self.sigma = sigma
        if f is not None:
            self.f = f
        else:
            self.f = GenericFunction(lambda x: np.random.rand(x.shape[0]), self.sigma)
        
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.init_samples = 20

    def update_posterior(self, x, y, copy=False): 

        if x.ndim==1:
            x=x[:,np.newaxis]


        f = CatBoostRegressor(iterations = 20,
                              random_seed=np.random.randint(100000),
                              verbose=False)
        if copy:
            Xtrain = np.concatenate((self.Xtrain, x))
            Ytrain = np.concatenate((self.Ytrain, y))
            if self.Xtrain.shape[0] < self.init_samples:
                f = GenericFunction(lambda x: self.sigma*np.random.rand(x.shape[0]), 
                                    self.sigma) 
                return Kitten(Xtrain, Ytrain, f) 
            else:
                f.fit(Xtrain, Ytrain)
                f = GenericFunction(lambda x: f.predict(x), self.sigma)
            return Kitten(Xtrain, Ytrain, f) 
        else:
            self.Xtrain = np.concatenate((self.Xtrain, x))
            self.Ytrain = np.concatenate((self.Ytrain, y))
            if self.Xtrain.shape[0] < 10:
                return
            f.fit(self.Xtrain, self.Ytrain)
            self.f = GenericFunction(lambda x: f.predict(x), self.sigma)
            
    
    def sample(self, k=1):
        if  self.Xtrain.shape[0] < self.init_samples:
            effs = [GenericFunction(lambda x: self.sigma*np.random.rand(x.shape[0]), 
                                    self.sigma) 
                    for i in range(k)]
        else:
            effs = []
            for i in range(k):        
                model = CatBoostRegressor(iterations = 20,
                                          random_seed=np.random.randint(100000),
                                          verbose=False)
                n = len(self.Xtrain)
                idx = np.random.choice(n,n)
                try:
                    f_tilde = model.fit(self.Xtrain[idx], self.Ytrain[idx]+np.random.randn(n)*self.sigma)
                    effs.append(GenericFunction(lambda x: f_tilde.predict(x), self.sigma))
                except:
                    print('excepted', self.Xtrain)
                    raise

        
        if k==1:
            return effs[0]
        return effs

    def map(self):
        # returns map estimate
        return self.f
    
    def identical(self):
        if np.all(np.isclose(self.Xtrain - self.Xtrain[0],0)):
            return True
        else:
            return False