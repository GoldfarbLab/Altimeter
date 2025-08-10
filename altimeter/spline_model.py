import torch
from torch import nn
import lightning as L
from .bspline import eval_bspline

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
        
        if inpce.dim() == 0:
            inpce = inpce.unsqueeze(0)
        inpce = inpce.unsqueeze(1).repeat(1, coefficients.shape[-1])
        
        out = eval_bspline(inpce, knots, coefficients, 3)
        
        return out