from time import time
import numpy as np

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
        
    def predict(self):
        pass