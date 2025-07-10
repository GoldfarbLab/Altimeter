verbose=False # print out intialization statistics for various tensors.

import torch
from torch import nn
import numpy as np
from typing import Optional


# NOTE: Target sigma
# sigma(a(0,siga).dot(b(0,sigb))) = siga*sigb*(dot_dim_len)**0.5
# try to aim for small std on residual (~ <=0.1) with sig_inp=1

"""
Initializers for various model tensors
- Very ad hoc for head(16,16,64), embedsz=256. If these parameters change,
  may need to change the initializers somewhat for optimal training.
"""
nm = 0.03
initEmb = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, nm)
initPos = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, nm)
initQ = lambda shps: (
    nn.init.normal_(torch.empty(shps), 0.0, shps[0]**-0.5*shps[1]**-0.25)
)
initK = lambda shps: (
    nn.init.normal_(torch.empty(shps), 0.0, shps[0]**-0.5*shps[1]**-0.25)
)
initL = lambda shps, eltargstd=0.5: (
    nn.init.normal_(torch.empty(shps), 0.0, eltargstd*shps[0]**-0.5)
)
initW = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, 0.01)
initV = lambda shps, wtargstd=0.012, sigWw=1, seq_len=40: (
    nn.init.normal_(torch.empty(shps), 0.0, 
    wtargstd**-1 * 
    shps[-1]**-0.5 * 
    shps[0]**-0.5 * 
    seq_len**-0.5 * 
    sigWw**-1)
)
initO = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, nm)
initFFN = lambda shps, stdin=1, targ=1: (
    nn.init.normal_(torch.empty(shps), 0.0, targ*shps[0]**-0.5**stdin**-1)
)
initProj = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, shps[0]**-0.5)
initFin = lambda tsr: nn.init.normal_(tsr, 0.0, .1)
# initFin = lambda tsr: nn.init.xavier_normal_(tsr)
initKnots = lambda shps: nn.init.constant_(torch.empty(shps), 0.85)
#initKnots = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, shps**-0.5)

class TrainableEltwiseBiasLayer(nn.Module):
    def __init__(self, n):
        super(TrainableEltwiseBiasLayer, self).__init__()
        self.weights = nn.Parameter(initKnots(n))  # define the trainable parameter
        
    def forward(self, x:torch.Tensor):
        return self.weights

class TalkingHeads(nn.Module):
    def __init__(self,
                 in_ch,
                 dk,
                 dv,
                 hv,
                 hk=None,
                 h=None,
                 drop=0,
                 out_dim=None
                 ):
        """
        Talking heads attention - Same as multi-headed attention, but for two 
        linear transformations amongst head dimensions before and after softmax.

        Parameters
        ----------
        in_ch : Dimension 1 of incoming tensor to the layer
        dk : Number of channels for queries and keys tensors
        dv : Number of channels in values tensor, projecting from softmax layer
        hv : Heads in values tensor
        hk : Heads in queries and keys tensors. None defaults to hv value
        h : Heads in softmax layer. None defaults to hv value
        drop : Dropout rate for residual layer
        out_dim : Channels in the output. None defaults to in_ch

        """
        super(TalkingHeads, self).__init__()
        self.dk = dk
        self.dv = dv
        self.hv = hv
        self.hk = hv if hk==None else hk
        self.h = hv if h==None else h
        self.out_dim = in_ch if out_dim==None else out_dim
        self.shortcut = (nn.Linear(in_ch, self.out_dim, bias=False) 
                         if self.out_dim!=in_ch else 
                         nn.Identity()
        )
        
        self.Wq = nn.Parameter(initQ((in_ch, dk, self.h)), requires_grad=True)
        #self.bq = nn.Parameter(torch.zeros(dk, self.h), requires_grad=True)
        self.Wk = nn.Parameter(initK((in_ch, dk, self.h)), requires_grad=True)
        #self.bk = nn.Parameter(torch.zeros(dk, self.h), requires_grad=True)
        self.Wv = nn.Parameter(initV((in_ch, dv, self.h)), requires_grad=True)
        #self.bv = nn.Parameter(torch.zeros(dv, self.h), requires_grad=True)
        self.alphaq = nn.Parameter(
            nn.init.constant_(torch.empty(1,), 2.), requires_grad=True
        ) # 2.
        self.alphak = nn.Parameter(
            nn.init.constant_(torch.empty(1,), 2.), requires_grad=True
        ) # 2.
        self.alphav = nn.Parameter(
            nn.init.constant_(torch.empty(1,), 2.), requires_grad=True
        ) # 2.
        
        self.Wl = nn.Parameter(initL((self.hk, self.h)), requires_grad=True)
        self.Ww = nn.Parameter(initW((self.h, self.hv)), requires_grad=True)
        
        self.Wo = nn.Parameter(initO((dv*self.h, self.out_dim)), requires_grad=True)
        #self.bo = nn.Parameter(torch.zeros(self.out_dim), requires_grad=True)
        self.drop = nn.Identity() if drop==0 else nn.Dropout(drop)
        
        if verbose:
            print("Wq=%f"%self.Wq.std())
            print("Wk=%f"%self.Wk.std())
            print("Wv=%f"%self.Wv.std())
            print("Wl=%f"%self.Wl.std())
            print("Ww=%f"%self.Ww.std())
            print("Wo=%f"%self.Wo.std())
    def forward(self, inp:torch.Tensor, mask:torch.Tensor):
        """
        Talking heads forward function

        Parameters
        ----------
        inp : Input tensor on which to perform attention. 
              inp.shape = (bs, in_ch, seq_len)
        mask : Vector that masks out null amino acids from softmax
               calculation. Set to zeros tensor if masking is undesired.
               mask.shape = (bs, seq_len)

        Returns
        -------
        Returns inp + residual(inp). Best to initialize so that residual is
        very small.
        
        activations : Means and standard deviations of intermediate tensors.
                      Useful for diagnosing instabilities, especially in J layer.
        W : Attention maps, if you want to visualize them.

        """
        Q = torch.sigmoid(self.alphaq)*torch.einsum('abc,bde->adce', inp, self.Wq)
        K = torch.sigmoid(self.alphak)*torch.einsum('abc,bde->adce', inp, self.Wk)
        V = torch.sigmoid(self.alphav)*torch.einsum('abc,bde->adce', inp, self.Wv)  
        
        J = torch.einsum('abcd,abed->aced', Q, K)
        EL = torch.einsum('abcd,de->abce', J, self.Wl) - mask[:, None, :, None]
        W = torch.softmax(EL, dim=2) # %1 zeros
        U = torch.einsum('abcd,de->abce', W, self.Ww)
        O = torch.einsum('abcd,aecd->abed', U, V)
        O = O.reshape(O.shape[0], -1, self.dv*self.hv)
        resid = self.drop(torch.einsum('abc,cd->adb', O, self.Wo))
        
        INP = self.shortcut(inp.transpose(-1,-2)).transpose(-1,-2)
        output = INP + resid
        
        return output

class FFN(nn.Module):
    def __init__(self,
                 in_ch,
                 units=None,
                 embed=None,
                 learn_embed=True,
                 drop=0
                 ):
        """
        Feed forward network
        
        :param in_ch: Dimension 1 of incoming tensor.
        :param units: Intermediate units projected to before ReLU. If None,
                      then defaults to in_ch.
        :param embed: Incoming units for for tensor adding charge and 
                      collision energy embedding before ReLU. If None then no
                      embedding is added in between linear transofrmations.
        :param learn_embed: If embed==units, setting to False allows embedding
                            to be unlearned. 
        :param drop: Dropout rate for residual layer.
        
        """
        super(FFN, self).__init__()
        
        units = in_ch if units==None else units
        self.embed = embed
        self.learn_embed = learn_embed
        if (self.embed is not None) and (self.learn_embed == False):
            assert units==self.embed,(
                "units must be equal to embed dimension if embed not learned"
            )
        
        self.W1 = nn.Parameter(initFFN((in_ch, units), 1, 1))
        self.W2 = nn.Parameter(initFFN((units, in_ch), 1, 0.1))
        if self.embed is not None:
            self.chce =  (
                nn.Linear(self.embed, units)
                if learn_embed | embed!=units else
                nn.Identity()
            )
        
        self.drop = nn.Identity() if drop==0 else nn.Dropout(drop)
        
        if verbose:
            print("FFN=%f"%self.W1.std())
    
    def forward(self, inp:torch.Tensor, embinp:torch.Tensor):
        """
        Feed forward network forward function.

        Parameters
        ----------
        inp : Input tensor. inp.shape = (bs, in_ch, seq_len)
        embinp : Embedded input of charge and collision energy fourier features.
                 embinp.shape = (bs, embed)

        Returns
        -------
        Returns inp + residual(inp)
        
        activations : Means and standard deviations of intermediate tensors.
                      Useful for diagnosing instabilities.

        """
        emb = self.chce(embinp).unsqueeze(-1)
        resid1 = torch.relu(torch.einsum('abc,bd->adc', inp, self.W1) + emb)
        resid2 = torch.einsum('abc,bd->adc', resid1, self.W2)
        resid2 = self.drop(resid2)
        
        output = inp + resid2
    
        return output

class TransBlock(nn.Module):
    def __init__(self,
                 hargs,
                 fargs
                 ):
        """
        Transformer block that implements attention and FFN layers with
        BatchNorm1d following each layer.
        
        :param hargs: Attention layer positional arguments.
        :param fargs: FFN layer positional arguments.
        """
        
        super(TransBlock, self).__init__()
        units1 = hargs[0] if hargs[-1]==None else hargs[-1]
        self.norm1 = nn.BatchNorm1d(units1)
        self.head = TalkingHeads(*hargs)
        self.norm2 = nn.BatchNorm1d(units1)
        self.ffn = FFN(*fargs)
        
    def forward(self, inp:torch.Tensor, embed:torch.Tensor, mask:torch.Tensor):
        """
        Forward call for TransBlock

        Parameters
        ----------
        inp : Input tensor. inp.shape = (bs, in_ch, seq_len)
        embed : Optional embedding tensor for charge and collision energy, fed 
                into FFN layer.
        mask : Optional mask for attention layer.

        Returns
        -------
        out : Output following Attention-BatchNorm1d-FFN-BatchNorm1d
        out2+out3 : tuple of intermediate activation statistics.
        FM : Feature/attention maps from attention layer.

        """
        out = self.head(inp, mask)
        out = self.norm1(out)
        out = self.ffn(out, embinp=embed)
        out = self.norm2(out)
        return out


class FlipyFlopy(nn.Module):
    def __init__(self,
                 in_ch=40,
                 seq_len=40,
                 out_dim=20000,
                 embedsz=256,
                 blocks=9,
                 head=(64,64,4),
                 units=None,
                 drop=0,
                 filtlast=512,
                 mask=False,
                 CEembed=False,
                 CEembed_units=256,
                 learn_ffn_embed=True,
                 pos_type='learned',
                 coefs=4,
                 knots=[1, 7, 13, 19, 25, 31, 37, 43],
                 num_fixed_knots=4
                 ):
        """
        Talking heads (Making Flippy Floppy) attention model

        Parameters
        ----------
        in_ch : Number of input channels for input embedding of at least 1)
                amino acid sequence, 2) modifications, 3-optional) charge,
                4-optional) collision energy.
        seq_len : Maximum sequence length for peptide sequence.
        out_dim : Output units for 1d output vector.
        embedsz : Channels for first linear projection, and likely running width
                  of network throughout.
        blocks : Number of TransBlocks to include in depth of network.
        head : Tuple of positional arguments for attention layer.
        units : Channels for first linear projection in FFN layer
        drop : Dropout rate to use in attention and FFN layers
        filtlast : Channels in penultimate layer.
        mask : Optional mask for attention head.
        CEembed : Option for embedding charge and energy with fourier features.
        CEembed_units : Number of fourier features to expand both charge and
                        energy by, before concatenation and feeding to FFN.
        learn_ffn_embed: If True, FFN layers project incoming CE embedding. If
                         False, 1 projections after concatenation, used throughout.
        pos_type : Either 'learned' positional embedding, otherwise fourier 
                   feature embedding.

        """
        super(FlipyFlopy, self).__init__()
        units = embedsz if eval(units)==None else units
        
        self.mask = mask
        self.out_dim = out_dim
        self.num_coefs = coefs
        self.knots = knots
        self.num_fixed_knots = num_fixed_knots
        
        self.embed = nn.Parameter(initEmb((in_ch, embedsz)), requires_grad=True)
        if pos_type=='learned':
            self.pos = nn.Parameter(initPos((embedsz, seq_len)), requires_grad=True)
        else:
            pos = (
                torch.arange(seq_len)[:,None] * 
                torch.exp(-np.log(1000) * 
                       torch.arange(embedsz//2)/(embedsz//2)
                )[None]
            )
            self.pos = nn.Parameter(
                torch.tensor(
                    torch.concatenate([torch.cos(pos),torch.sin(pos)], axis=-1).T[None], 
                    dtype=torch.float32
                ), requires_grad=False
            )
        self.embed_norm = nn.BatchNorm1d(embedsz)
        
        self.CEembed = CEembed
        self.cesz = CEembed_units
        if CEembed:
            self.denseCH = nn.Linear(self.cesz, self.cesz)
            self.denseCE = nn.Linear(self.cesz, self.cesz)
            #self.postcat = (nn.Identity() if learn_ffn_embed else 
            #                 nn.Linear(2*self.cesz, units)
            #)
            self.postcat =  nn.Linear(self.cesz, units)
            ffnembed = 2*CEembed_units if learn_ffn_embed else units
        else:
            ffnembed = None
        
        head_args = (embedsz,)+tuple(head)+(None,None,drop,None)
        ffn_args = (embedsz, units, ffnembed, learn_ffn_embed, drop)
        self.main = nn.ModuleList([
            TransBlock(head_args, ffn_args) 
            for _ in range(blocks)]
        )
        
        self.Proj = nn.Parameter(initProj((embedsz, filtlast)), requires_grad=True)
        self.ProjNorm = nn.BatchNorm1d(filtlast)
        

        self.final = nn.Sequential(nn.Linear(filtlast, out_dim*self.num_coefs), nn.Sigmoid())
        
        ########################
        # Knot layer
        self.knot_linear = TrainableEltwiseBiasLayer(len(self.knots) - self.num_fixed_knots) 
        
        
        ########################
        
        initFin(self.final[0].weight)
        nn.init.zeros_(list(self.final.parameters())[1])
        
        self.global_step = nn.Parameter(torch.tensor(0), requires_grad=False)
        
        if verbose:
            print("Embed: %f"%self.embed.std())
            print("Pos: %f"%self.pos.std())
            print("Proj: %f"%self.Proj.std())
            print("Final: %f"%list(self.final.parameters())[0].std())
    
    def embedCE(self, ce, embedsz:int, freq:float=10000.):
        """
        Generating fourier features for either charge and energy or positional
        embedding.

        Parameters
        ----------
        ce : Charge, energy, or positional value
        embedsz : Number of features to expand ce by.
        freq: Frequency parameter.

        Returns
        -------
        bs x embedsz fourier feature tensor.

        """
        # ce.shape = (bs,)
        embed = (
            ce[:,None] * 
            torch.exp(
                -torch.log(torch.tensor(freq)) * 
                torch.arange(embedsz//2, device=ce.device)/(embedsz//2)
            )[None]
        )
        return torch.cat([torch.cos(embed),torch.sin(embed)],dim=1)      
    
    def total_params(self, trainable=True):
        """
        Count the total number of model parameters.

        Parameters
        ----------
        trainable : Only count trainable parameters.

        """
        if trainable:
            print(sum([np.prod(m.shape) for m in list(self.parameters()) 
                       if m.requires_grad==True]))
        else:
            print(sum([np.prod(m.shape) for m in list(self.parameters())]))
            
    def forward_coef(self, inp:tuple[torch.Tensor, torch.Tensor]):
        inp, inpch = inp
    #def forward(self, inp:torch.Tensor, inpch:torch.Tensor):
   

        if len(inpch.shape) == 0:
            inpch = inpch.unsqueeze(-1)
        elif len(inpch.shape) == 2:
            inpch = inpch.squeeze(1)
            
        ch_embed = nn.functional.silu(
            self.denseCH(self.embedCE(inpch, self.cesz, 10.0))
        )
        embed = self.postcat(torch.cat([ch_embed],-1))
        
        # inp.shape: bs, in_ch, seq_len
        mask = (1e5*inp[:,20] 
                if self.mask else 
                torch.zeros((inp.shape[0], inp.shape[-1]), device=inp.device)
        )
        
        out = torch.einsum('abc,bd->adc', inp, self.embed) # bs, embedsz, seq_len
        out += self.pos
        out = self.embed_norm(out)
        for layer in self.main:
            out = layer(out, embed, mask)
        out = torch.relu(self.ProjNorm(torch.einsum('abc,bd->adc', out, self.Proj))) # bs, filtlast, seq_len

        # get b-spline coefficients
        poly_coef = torch.reshape(self.final(out.transpose(-1,-2)).mean(dim=1), (inp.shape[0], self.num_coefs, self.out_dim))
        knots = self.get_knots().unsqueeze(0).repeat(inp.shape[0], 1)
        auc = self.integrate(knots[0], poly_coef)
        
        return poly_coef, knots, auc 
        
    def forward(self, inp):
        [inp, inpch, inpce] = inp
        poly_coef, knots, _ = self.forward_coef([inp, inpch])
             
        # create knots
        knots = knots.unsqueeze(2).repeat(1,1,self.out_dim)
        
        if len(inpce.shape) == 0:
            inpce = inpce.unsqueeze(-1)
        inpce = inpce.unsqueeze(1).repeat(1, self.out_dim)

        out = bspline(inpce, knots, poly_coef, 3)
        
        return out
    
    def get_knots(self):
        knots_left = torch.tensor(self.knots[:(self.num_fixed_knots >> 1)], device=self.knot_linear.weights.device)
        knots_center = torch.cumsum(torch.sigmoid(self.knot_linear.weights)*10, dim=0) + self.knots[1] + (self.knots[1]-self.knots[0])# 20 #0:2
        knots_right = torch.tensor(self.knots[-(self.num_fixed_knots >> 1):], device=self.knot_linear.weights.device)
        
        knots = torch.cat((knots_left, knots_center, knots_right))
        
        return knots
    
    # from NCE 20 to 40
    def integrate(self, knots, poly_coef):
        auc = torch.zeros(poly_coef.shape[0], poly_coef.shape[2], device=poly_coef.device)
        auc += poly_coef[:,0,:] * (knots[4] - knots[0]) * 0.08408505396482596 #0.0752889089763667 
        auc += poly_coef[:,1,:] * (knots[5] - knots[1]) * 0.47690670779403865 #0.3075992695693613 #
        auc += poly_coef[:,2,:] * (knots[6] - knots[2]) * 0.19068693745182305 #0.07144580350892756 #
        auc += poly_coef[:,3,:] * (knots[7] - knots[3]) * 0.00731285949764904  #0.0009773926870403494 #
        return auc / 4
            



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
def bspline(x, t, c, k):
    n = t.shape[1] - k - 1
    out = torch.zeros_like(x)
    for i in range(n):
        out += c[:, i, :] * B(x, k, i, t) 
    return out
