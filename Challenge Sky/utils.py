from time import time
from scipy.signal import convolve2d
from scipy.optimize import curve_fit, minimize
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

SOBEL = np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]])

def timeit(func):
    def new_func(*args, **kwargs):
        print(f"{func.__name__}", end='')
        time_init = time()
        res = func(*args, **kwargs)
        print(f" computed in : {time()-time_init:.2f}s.")
        return res
    return new_func

class ShifterPro():
    
    def __init__(self, image_ref):
        self.image_ref = image_ref
        
        dX = convolve2d(self.image_ref, SOBEL,   mode = "same")
        dY = convolve2d(self.image_ref, SOBEL.T, mode = "same")
        E = dX**2 + dY**2
        E = E[5:-5,5:-5]
        
        self.x_ref, self.y_ref = np.unravel_index(E.argmax(), E.shape) + 50*np.ones(2).astype(int)
    
    @timeit    
    def predict(self, image, r_disk, r_search):
        self.win_ref = self.image_ref[self.x_ref-r_disk:self.x_ref+r_disk,self.y_ref-r_disk:self.y_ref+r_disk]
        
        @np.vectorize
        def compute_value(i, j):
            i -= r_search-self.x_ref
            j -= r_search-self.y_ref
            i, j = int(i), int(j)          
            return np.linalg.norm(self.win_ref - image[i-r_disk:i+r_disk,j-r_disk:j+r_disk])
        
        @np.vectorize
        def compute_value_conv(i, j):
            i += self.x_ref - r_search
            j += self.y_ref - r_search
            i, j = int(i), int(j)          
            return np.linalg.norm(convolve2d(self.win_ref,image[i-r_disk:i+r_disk,j-r_disk:j+r_disk], mode = "same"))
        
        D = np.fromfunction(compute_value_conv, (2*r_search+1,2*r_search+1))
        
        def func_to_fit(data, a, b, c, d, e, f):
            x,y = data
            return a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
        
        x,y = np.meshgrid(range(2*r_search+1),range(2*r_search+1))
        
        args, _ = curve_fit(func_to_fit, (x.flatten(),y.flatten()), D.flatten())
        
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(D,cmap="viridis")
        
        ax[1].imshow(func_to_fit([x,y],*args),cmap="viridis")
        
        f = lambda x:func_to_fit(x,*args)
        return minimize(f, [10,10])
        
        
        
# 0.8670 0.2505
    
    
#dist(image, centre, rayondisque, rayonrecherche)
#Renvoie la matrice distance

# mini matrice