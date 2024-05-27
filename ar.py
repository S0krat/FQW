import numpy as np
from scipy.optimize import minimize

class Autoregression:
    def __init__(self, p: int):
        self.p = p
        self.coefs = None

    def fit(self, x):
        def wrapped_sos(c):
            return self.__sos(x, c)
        
        def wrapped_grad(c):
            return self.__sos_grad(x, c)
        
        hess = self.__sos_hess(x)
        def wr_hess(c):
            return hess
        
        x0 = np.zeros(self.p)
        res = minimize(wrapped_sos, x0, method='Newton-CG', jac=wrapped_grad, hess=wr_hess)
        if res.success:
            print("Success!")
        else:
            print("Оптимизация пошла по так называемой pizde")
        self.coefs = res.x

    def predict(self, a2d):
        res = []
        for a1d in a2d:
            res.append(self.__pred(a1d))
        return np.array(res)

    def __pred(self, a1d):
        return np.dot(self.coefs, a1d)

    def __sos(self, x, c):
        n = len(x)
        sos = 0
        for i in range(self.p, n, 7):
            sos += (x[i] - np.dot(x[i - self.p:i], c)) ** 2
        return sos
    
    def __sos_grad(self, x, c):
        n = len(x)
        m = len(c)
        grad = np.zeros(m)
        for i in range(self.p, n, 7):
            for j in range(m):
                grad[j] += -2 * x[i - self.p + j] * (x[i] - np.dot(x[i - self.p:i], c))
        return grad
    
    def __sos_hess(self, x):
        m = self.p
        n = len(x)
        hess = np.zeros((m, m))
        for i in range(self.p, n, 7):
            for j in range(m):
                for k in range(m):
                    hess[j, k] += 2 * x[i - self.p + j] * x[i - self.p + k]
        return hess
