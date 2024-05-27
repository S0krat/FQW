from pmdarima.arima import ARIMA
import numpy as np

class MyArma:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.intercept = None
        self.ar_coefs = None
        self.ma_coefs = None
        self.last_p = np.array([])
        self.last_q = np.array([])
    
    def fit(self, x):
        if self.p:
            self.last_p = x[-self.p:]
        model = ARIMA(order=(self.p, 0, self.q))
        model.fit(x)
        if self.q:
            self.last_q = model.resid()[-self.q:]
        params = np.flip(np.array(model.params()))
        self.intercept = params[-1]
        self.ma_coefs = params[1:self.q + 1]
        self.ar_coefs = params[self.q + 1:self.p + self.q + 1]
        
    def predict_next(self):
        return self.intercept + np.dot(self.ar_coefs, self.last_p) + np.dot(self.ma_coefs, self.last_q)
    
    def predict(self, n):
        result = []
        temp_last_p = self.last_p.copy()
        temp_last_q = self.last_q.copy()
        for _ in range(n):
            elem = self.intercept + np.dot(self.ar_coefs, temp_last_p) + np.dot(self.ma_coefs, temp_last_q)
            result.append(elem)
            temp_last_p = np.append(temp_last_p, [elem])[1:]
            temp_last_q = np.append(temp_last_q, [0])[1:]
        return result
    
    def update(self, ys):
        for y in ys:
            pred = self.predict_next()
            self.last_p = np.append(self.last_p, [y])[1:]
            self.last_q = np.append(self.last_q, [y - pred])[1:]


    