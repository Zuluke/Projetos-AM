import numpy as np
from sklearn.metrics import adjusted_rand_score
# from typing import 

class KFCM_K_W1:
    
    def __init__(self, n_clusters, n_iter, epsilon, m, verbose=True, random_state=42):
        self._n_clusters = n_clusters # parameter c
        self._n_iter = n_iter
        self._epsilon = epsilon
        self._s = None
        self._G = None
        self._U = None 
        
        self._n = None
        self._p = None
        self._m = m
        
        self.verbose = verbose

        if random_state:
            np.random.seed(random_state)
    
    def _initialize_U(self):
        # self._U = np.random.rand(self._n_samples, self._n_clusters)
        # self._U = self._U / np.sum(self._U, axis=1)[:, np.newaxis]
        self._U = np.zeros((self._n, self._n_clusters))

    def _initialize_s(self):
        self._s = np.ones(self._p)
    
    def _initialize_g(self):
        indexes = np.random.choice(self._n, self._n_clusters, replace=False) 
        self._G = self._X[indexes, :]
        #print("------ G ------")
        #print(self._G)
    @staticmethod
    def compute_kernel(X_l, X_k, s):
        
        X_l = X_l[:, np.newaxis, :] # (n, 1, p)
        X_k = X_k[np.newaxis, :, :] # (1, c, p)
        
        return np.exp(-0.5 * np.sum(s * (X_l - X_k)**2, axis=2))

    def _update_U(self):
        #print("------ UPDATE U ------")
        #print("------ G ------")
        #print(self._G)
        K = self.compute_kernel(self._X, self._G, self._s)
        #print("------ K ------")
        #print(K)
        
        numerator = 2 - 2 * K[:, :, np.newaxis] # (n, c, 1)
        numerator = np.maximum(numerator, 1e-20)
        denominator = 2 - 2 * K[:, np.newaxis, :] # (n, 1, c)
        denominator = np.maximum(denominator, 1e-20)
        #print("------ Denominator ------")
        #print(denominator)
        #print("------ Numerator ------")
        #print(numerator)
        
        
        ratio = (numerator / denominator) ** (1 / (self._m - 1))
        #print("------ U ------")
        #print(1/np.sum(ratio, axis=2))
        self._U = 1/np.sum(ratio, axis=2)
    
    
    def calculate_objective_function(self):
        K = self.compute_kernel(self._X, self._G, self._s)
        J = np.sum(self._U ** self._m * (2 - 2 * K))
        return J
    
    def _update_s(self):
        K = self.compute_kernel(self._X, self._G, self._s)
        #print("------ UPDATE S ------")
        numerator = 1
        
        for h in range(self._p):
            diff_squared = (self._X[:, h][:, np.newaxis] - self._G[:, h][np.newaxis, :]) ** 2
            #print('diff_squared', self._U ** self._m * K * diff_squared)
            total_sum = np.sum(self._U ** self._m * K * diff_squared)
            #print('total_sum', total_sum)
            numerator *= total_sum ** (1 / self._p)
        
        diff = (self._X[:, np.newaxis, :] - self._G[np.newaxis, :, :]) ** 2
        denominator = np.sum(self._U ** self._m * K * np.sum(diff, axis=2))
        
        # print(f'numerator = {numerator}')
        # print(f'denominator = {denominator}')
        self._s = numerator / np.maximum(denominator, 1e-20)
    
    def _update_G(self):
        K = self.compute_kernel(self._X, self._G, self._s)
        numerator = np.sum((self._U ** self._m)[:,:, np.newaxis] * K[:, :, np.newaxis] * self._X[:, np.newaxis, :], axis=0)
        denominator = np.sum(self._U ** self._m * K, axis=0)[:, np.newaxis]
        
        self._G = numerator / denominator
    
    
    def calculate_modified_partition_coefficient(self):
        pc = np.sum(self._U**2) / self._n
        mpc = 1 - (self._n_clusters / (self._n_clusters - 1)) * (1 - pc)
        return mpc

    def _evaluate_adjusted_rand_score(self):
        y_pred = np.argmax(self._U, axis=1)
        print(self._y.shape, y_pred.shape)
        return adjusted_rand_score(self._y, y_pred)

    def evaluate(self):
        return {
            "MPC":self.calculate_modified_partition_coefficient(),
            "rand": self._evaluate_adjusted_rand_score()
        }
        
    
    def fit(self, X, y=None):
        from sklearn.preprocessing import MinMaxScaler
        self._n, self._p = X.shape
        scaler = MinMaxScaler()
        self._X = scaler.fit_transform(X)
        self._y = y
        
        ## Initialization
        self._initialize_U()
        self._initialize_s()
        self._initialize_g()
        self._update_U()
        #print("------ BEGIN ------")
        J_new = self.calculate_objective_function()
        
        for t in range(self._n_iter):
            J_old = J_new
                        
            # Computation of the width parameters
            self._update_s()
            
            # Computation of the fuzzy cluster prototypes
            self._update_G()
            
            # Computation of the membership degrees
            self._update_U()
            
            J_new = self.calculate_objective_function()
            
            if self.verbose:
                print(f'Iteration {t+1}: J = {J_new}')
                print(self.evaluate())    

            if np.abs(J_new - J_old) < self._epsilon:
                break

                
        