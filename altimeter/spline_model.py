import torch
from torch import nn
import numpy as np
import lightning as L

class LitBSplineNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BSplineNN()
        
    def forward(self, coefficients, knots, inpce):
        return self.model(coefficients, knots, inpce)
         

class BSplineNN(nn.Module):
    def __init__(self):
        super(BSplineNN, self).__init__()
    
    def forward(self, coefficients, knots, inpce):
        # create knots
        knots = knots.unsqueeze(2).repeat(1, 1, coefficients.shape[-1])
        
        inpce = inpce.repeat(1, coefficients.shape[-1])

        out = bspline(inpce, knots, coefficients, 3)
        
        return out

# x = NCEs
# k = polynomial degree
# i = basis-spline index
# t = knots
def B(x, k, i, t):
    out = torch.zeros_like(x)
    
    if k == 0:
        out = torch.where(torch.logical_and(t[:,i,:] <= x, x < t[:,i+1,:]), 1.0, 0.0)
        return out

    if t[0, i+k, 0] == t[0, i, 0]:
        c1 = torch.zeros_like(x)
    else:
        c1 = (x - t[:,i,:])/(t[:,i+k,:] - t[:,i,:]) * B(x, k-1, i, t)
        
    if t[0, i+k+1, 0] == t[0, i+1, 0]:
        c2 = torch.zeros_like(x)
    else:
        c2 = (t[:,i+k+1,:] - x)/(t[:,i+k+1,:] - t[:,i+1,:]) * B(x, k-1, i+1, t)
    
    return c1 + c2

# x = NCEs
# t = knots
# c = coefficients
# k = polynomial degree
def bspline(x:torch.Tensor, t:torch.Tensor, c:torch.Tensor, k:int):
    n = t.shape[1] - k - 1
    out = torch.zeros_like(x)
    for i in range(n):
        out += c[:, i, :] * B(x, k, i, t) 
    return out