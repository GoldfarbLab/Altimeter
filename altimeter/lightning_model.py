import lightning as L
import torch as torch
import sys
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import wandb
from models_spline import FlipyFlopy
from plot_utils import scoreDistPlot

# define the LightningModule
class LitFlipyFlopy(L.LightningModule):
    """Lightning wrapper around :class:`FlipyFlopy`."""

    def __init__(self, config, model_config):
        super().__init__()
        self.model = FlipyFlopy(**model_config)
        self.config = config
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss = self._compute_train_loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.config['batch_size'],
        )
        for i, x in enumerate(self.model.get_knots().detach().cpu().numpy()):
            self.log(
                "knots" + str(i),
                x,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=self.config['batch_size'],
            )
        return loss
    
    def validation_step(self, batch, batch_idx):
        losses = self._compute_eval_loss(batch)
        self.log(
            "val_SA_mean",
            losses.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.config['test_batch_size'],
        )
        self.validation_step_outputs.append(losses)
        return losses
    
    def test_step(self, batch, batch_idx):
        losses = self._compute_eval_loss(batch)
        self.log(
            "test_SA_mean",
            losses.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.config['test_batch_size'],
        )
        return losses
    
    def predict_step(self, batch, batch_idx):
        losses = self._compute_eval_loss(batch)
        for i, loss in enumerate(losses):
            print("figure", i, loss.item(), len(batch["seq"][i]))
        return losses
    
    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)
        self.log("val_SA_median", all_preds.median(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        scoreDistPlot(all_preds, "val", self.logger)
        self.validation_step_outputs.clear()  # free memory
    
    def forward(self, x:tuple[torch.Tensor, torch.Tensor]):
        return self.model(x)
    
    #def forward(self, x:torch.Tensor, y:torch.Tensor):
    #    return self.model(x,y)
    
    def forward_coef(self, x):
        return self.model.compute_coefficients(x)
    
    
    
    def _forward_batch(self, batch):
        samples = batch['samples']
        targ = batch['targ']
        mask = batch['mask']
        LODs = batch['LOD']
        weights = batch['weight']
        mask_zero = batch['min_mz'] == "0"
        out = self.model(samples)
        return targ, mask, LODs, weights, mask_zero, out

    def _compute_train_loss(self, batch):
        targ, mask, LODs, weights, mask_zero, out = self._forward_batch(batch)
        return LossFunc(targ, out, mask, LODs, weights, root=self.config['root_int'], mask_zero=mask_zero)

    def _compute_eval_loss(self, batch):
        targ, mask, LODs, weights, mask_zero, out = self._forward_batch(batch)
        return -LossFunc(targ, out, mask, LODs, weights, root=self.config['root_int'], do_mean=False, do_weights=False, mask_zero=mask_zero)


    


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), eval(self.config['lr']))
        scheduler = ExponentialLR(optimizer, gamma=self.config['lr_decay_rate'])
        return [optimizer], [scheduler]
    
    ###############################################################################
    ############################# Visualization ###################################
    ###############################################################################
    
    
###############################################################################
############################# Loss function ###################################
###############################################################################
CS = torch.nn.CosineSimilarity(dim=-1)
KL = torch.nn.KLDivLoss()
CE = torch.nn.CrossEntropyLoss()
MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()

def LossFunc(targ, pred, mask, LOD, weights, root, doFullMask=True, epsilon=1e-5, do_weights=True, do_mean=True, mask_zero=False): 
    #pred /= torch.max(pred, dim=1, keepdim=True)[0]
    #pred /= torch.sum(pred, dim=1, keepdim=True)[0]
    
    targ = torch.squeeze(targ, 1)
    mask = torch.squeeze(mask, 1)
    
    if mask_zero:
        targ, pred = apply_mask_zero(targ, pred)
    else:
        targ, pred = apply_mask(targ, pred, mask, LOD, doFullMask)
    
    targ = root_intensity(targ, root=root) if root is not None else targ
    pred = root_intensity(pred, root=root) if root is not None else pred
    
    cs = CS(targ, pred)
    cs = torch.clamp(cs, min=-(1-epsilon), max=(1-epsilon))
    sa = -(1 - 2 * (torch.arccos(cs) / torch.pi))

    if torch.any(torch.isnan(sa)):
        print("nan unweighted SA")
        sys.exit()
    
    if do_weights:
        weighted = sa * weights
        weighted = weighted.sum() / weights.sum()
        return weighted
    
    if do_mean:
        sa = sa.mean()
    
    return sa


def LossFuncEuc(targ, pred, mask, LOD, weights, root, doFullMask=True, epsilon=1e-5, do_weights=True, do_mean=True, mask_zero=False): 
    #pred /= torch.max(pred, dim=1, keepdim=True)[0]
    #pred /= torch.sum(pred, dim=1, keepdim=True)[0]
    
    targ = torch.squeeze(targ, 1)
    mask = torch.squeeze(mask, 1)
    
    if mask_zero:
        targ, pred = apply_mask_zero(targ, pred)
    else:
        targ, pred = apply_mask(targ, pred, mask, LOD, doFullMask)
    
    targ = root_intensity(targ, root=root) if root is not None else targ
    pred = root_intensity(pred, root=root) if root is not None else pred
    
    
    #loss = CE(pred, targ)
    #loss = torch.mean((pred - targ)**2)
    
    loss = L1(pred, targ)
    #loss = torch.sqrt(torch.clamp((pred - targ)**2, min=1e-20)).sum(dim=1)
    #cs = CS(targ, pred)
    #cs = torch.clamp(cs, min=-(1-epsilon), max=(1-epsilon))
    #sa = -(1 - 2 * (torch.arccos(cs) / torch.pi))

    if torch.any(torch.isnan(loss)):
        print("nan unweighted MSE")
        sys.exit()
    
    if do_weights:
        weighted = loss * weights
        weighted = weighted.sum() / weights.sum()
        return weighted
    
    if do_mean:
        loss = loss.mean()
    
    return loss

def apply_mask_zero(targ, pred):
    ###pred = torch.where(targ==0.0, 0.0, pred)
    pred = torch.where(torch.logical_and(targ==0.0, pred>0), 0.0, pred)
    return targ, pred
    

def apply_mask(targ, pred, mask, LOD, doFullMask=True):
    #LOD = torch.reshape(LOD, (LOD.shape[0], 1, 1)).expand_as(targ)
    LOD = torch.reshape(LOD, (LOD.shape[0], 1)).expand_as(targ)
    
    # mask below limit of detection
    ###pred = torch.where(torch.logical_and(targ==0, pred<=LOD), 0.0, pred)  
    pred = torch.where(torch.logical_and(targ==0, torch.logical_and(pred<=LOD, pred>0)), 0.0, pred)  
    if doFullMask:
        pred = torch.where(torch.logical_and(targ==0, pred>LOD), pred-LOD, pred)

    # mask 1 - outside of scan range. Can have any intensity without penalty
    ###pred = torch.where(mask==1, 0.0, pred)
    pred = torch.where(torch.logical_and(mask==1, pred>0), 0.0, pred)
    targ = torch.where(mask==1, 0.0, targ)
    
    # mask 2-5 - bad isotope dist, below purity, high m/z error, ambiguous annotation. Can have any intensity up to the target
    if doFullMask:
        ###pred = torch.where(torch.logical_and(mask>1, pred < targ), 0.0, pred)
        pred = torch.where(torch.logical_and(mask>1, torch.logical_and(pred < targ, pred>0)), 0.0, pred)
        pred = torch.where(torch.logical_and(mask>1, pred > targ), pred-targ, pred)
        targ = torch.where(mask>1, 0.0, targ)
        
        #pred = torch.where(torch.logical_and(torch.logical_and(mask>1, mask !=3), pred < targ), 0.0, pred)
        #pred = torch.where(torch.logical_and(torch.logical_and(mask>1, mask !=3), pred > targ), pred-targ, pred)
        #targ = torch.where(torch.logical_and(mask>1, mask !=3), 0.0, targ)
        
    return targ, pred
    
def root_intensity(ints, root=2):
    if root==2:
        ints[ints>0] = torch.sqrt(ints[ints>0]) # faster than **(1/2)
    else:
        ints[ints>0] = ints[ints>0]**(1/root)
    return ints



