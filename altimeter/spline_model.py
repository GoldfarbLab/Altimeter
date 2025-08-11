import torch
from torch import nn
import lightning as L
from bspline import eval_bspline

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
        # match knot tensor to the output dimension of the coefficients
        knots = knots.unsqueeze(2).repeat(1, 1, coefficients.shape[-1])

        # ensure the collision energy tensor has shape (batch, out_dim)
        if inpce.dim() == 0:
            # scalar collision energy
            inpce = inpce.view(1, 1)
        elif inpce.dim() == 1:
            # shape (batch,) -> (batch,1)
            inpce = inpce.unsqueeze(1)
        elif inpce.dim() == 2:
            # already (batch,1)
            pass
        else:
            raise RuntimeError("inpce must have 0, 1, or 2 dimensions")

        inpce = inpce.expand(-1, coefficients.shape[-1])

        out = eval_bspline(inpce, knots, coefficients, 3)

        return out