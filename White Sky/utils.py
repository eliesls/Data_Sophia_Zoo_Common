from time import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, recall_score

def timeit(func):
    def new_func(*args, **kwargs):
        print(f"{func.__name__}", end='')
        time_init = time()
        res = func(*args, **kwargs)
        print(f" computed in : {time()-time_init:.2f}s.")
        return res
    return new_func

def evaluate(model, X, Y_true):
    nb_sample = X.shape[0]
    Y_pred = model.predict(X)
    print(f"Accuracy  : {accuracy_score(Y_true, Y_pred):.2f}")
    print(f"F1 score  : {f1_score(Y_true, Y_pred):.2f}")
    print(f"Precision : {average_precision_score(Y_true, Y_pred):.2f}")
    print(f"Recall    : {recall_score(Y_true, Y_pred):.2f}")
    
class LDA():
    
    def __init__(self):
        self.w_dict = None
    
    @timeit    
    def fit(self, X, Y):
        y_list = np.unique(Y)
        
        x_class      = {y:X[Y==y,:] for y in y_list}
        self.x_covs  = np.cov(x_class[y_list[0]].T)
        self.x_means = {y:np.mean(x_class[y], axis = 0) for y in y_list}
        self.x_prob  = {y:x_class[y].shape[0]/X.shape[0] for y in y_list}
        return 0
    
    @timeit    
    def predict(self, X):
        assert self.x_prob != None
        
        pred_list = list()
        for x in X:
            scores = [ 2*self.x_means[y].T @ np.linalg.pinv(self.x_covs) @ x 
                     - self.x_means[y].T @ np.linalg.pinv(self.x_covs) @ self.x_means[y] 
                     + 2 * np.log(self.x_prob[y]) for y in self.x_means]          
            y = np.argmax(scores)
            pred_list.append(y)
        return np.array(pred_list)
        
class QDA():
    
    def __init__(self):
        pass
        
    def fit(self, X, Y):
        y_list = np.unique(Y)
        
        x_class = {y:X[Y==y] for y in y_list}
        self.x_covs = {y:np.cov(x_class[y].T) for y in y_list}
        self.x_means = {y:np.mean(x_class[y], axis = 0) for y in y_list}
        self.x_prob  = {y:x_class[y].shape[0]/X.shape[0] for y in y_list}
        return 0
    
    @timeit    
    def predict(self, X):
        assert self.x_prob != None
        
        pred_list = list()
        for x in X:
            scores = [ (x-self.x_means[y]) @ np.linalg.pinv(self.x_covs[y]) @ (x-self.x_means[y])
                     + np.log(np.linalg.det(self.x_covs[y]))
                     - 2 * np.log(self.x_prob[y]) for y in self.x_means]          
            y = np.argmin(scores)
            pred_list.append(y)
        return np.array(pred_list)