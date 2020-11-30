from time import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, recall_score
import matplotlib.pyplot as plt
import os

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
    print(f"Accuracy  : {accuracy_score(Y_true, Y_pred):.3f}")
    print(f"F1 score  : {f1_score(Y_true, Y_pred):.3f}")
    print(f"Precision : {average_precision_score(Y_true, Y_pred):.3f}")
    print(f"Recall    : {recall_score(Y_true, Y_pred):.3f}")
    
def load_images():
    ls_data = os.listdir("Data/")

    list_label = [lbl[4:-4] for lbl in ls_data if lbl[:4] == "ima_"]

    X_list = [plt.imread(f"Data/ima_{label}.jpg") for label in list_label]
    Y_list = [plt.imread(f"Data/mask_{label}_skymask.png") for label in list_label]

    return X_list, Y_list

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
    
class Kernel():
    
    def __init__(self):
        pass
        
    @timeit    
    def fit(self, X, Y, var):
        self.X = X
        self.Y = Y
        self.var = var
        
    @timeit    
    def predict(self, X):
        pred = list()
        for x in X:
            prob_y = np.sum(self.Y*np.exp(-np.linalg.norm(x-self.X, axis = 1)**2/(2*self.var)))/self.Y[self.Y==1].size
            prob_x = np.sum((1-self.Y)*np.exp(-np.linalg.norm(x-self.X, axis = 1)**2/(2*self.var)))/self.Y[self.Y==0].size
            pred.append(1 if prob_y>prob_x else 0)
        return pred