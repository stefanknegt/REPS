import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
import sys

def logsumexponent(term,N):
    '''
    This is a trick to avoid overflow in the log.
    '''
    max_val = torch.max(term)
    term_corrected = term - max_val
    sumOfExp = torch.sum(torch.exp(term_corrected) / N)
    return max_val + torch.log(sumOfExp)

def check_values(*args,**kwargs):
    "Function to check where NaN values arise"
    nan=False
    for k,v in kwargs.items():
        if type(v) is np.ndarray:
            if np.isnan(v.any()):
                print("NAN DETECTED FOR:",k)
                nan=True
        else:
            if torch.isnan(torch.sum(v)):
                print("NAN DETECTED FOR:",k)
                nan=True

    if nan == True:
        for k,v in kwargs.items():
            print(k,v)
        sys.exit()
    return 0
