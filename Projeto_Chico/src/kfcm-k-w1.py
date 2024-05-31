import numpy as np
# from typing import 

class KFCM_K_W1:
    
    def __init__(self, n_clusters, n_iter, epsilon, random_state=42):
        self._n_clusters = n_clusters
        self._n_iter = n_iter
        self._epsilon = epsilon
        self._s = None
        self._G = None
        self._U = None 
        
        self._n = None
        self._p = None
        self._random_state = random_state
        
        if self._random_state:
            np.random.seed(self._random_state)
    
    def _initialize_U(self):
        self._U = np.random.rand(self._n_samples, self._n_clusters)
        self._U = self._U / np.sum(self._U, axis=1)[:, np.newaxis]
    
    def _initialize_s(self):
        self._s = np.ones(self._p)
    
    def _initialize_g(self):
        indexes = np.random.choice(self._n, self._n_clusters, replace=False) 
        self._G = self._X[indexes, :]
    
    @staticmethod
    def compute_kernel(X_l, X_k, s):
        return np.exp(-0.5 * np.sum(s * (X_l - X_k )**2, axis=1))
    
    def _update_U(self):
        # self._U = np.sum(
        #     (2 - 2 * self.compute_kernel(self._X, self._G, self._s)) /
        #     (2 - 2 * self.co)
        #     axis=1
        # )
        pass
    def fit(self, X):
        self._n, self._p = X.shape

        ## Initialization
        self._initialize_U()
        self._initialize_s()
        self._initialize_g()
        
                
        