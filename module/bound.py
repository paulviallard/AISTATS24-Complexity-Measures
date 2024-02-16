import torch
import numpy as np
import math
from core.kl_inv import kl_inv
from module.module import Module

###############################################################################


class MAE(Module):

    def forward(self, y_pred, y):
        ref = y_pred
        y_pred, y = self.to_torch(y_pred, y)

        loss = torch.abs(y_pred-y)
        loss = torch.mean(loss)

        return self.from_torch(ref, loss)

###############################################################################


class ZeroOne(Module):

    def forward(self, y_pred, y):
        ref = y_pred
        y_pred, y = self.to_torch(y_pred, y)

        _, y_pred = torch.max(y_pred, axis=1)
        y_pred = y_pred.unsqueeze(1)
        loss = torch.mean((y_pred != y).float())
        return self.from_torch(ref, loss)


###############################################################################


class GapBoundComp(Module):

    def forward(
        self, comp_prior_prior, comp_prior_post,
        comp_post_prior, comp_post_post,
        m, delta
    ):
        ref = comp_prior_prior
        (comp_prior_prior, comp_prior_post, comp_post_prior, comp_post_post,
         m, delta) = self.to_torch(
             comp_prior_prior, comp_prior_post,
             comp_post_prior, comp_post_post, m, delta)

        comp = (comp_prior_post-comp_prior_prior)
        comp = comp - (comp_post_post-comp_post_prior)
        bound = (1/m)*(
            comp + math.log((2*math.sqrt(m))/((0.5*delta)**2.0)))

        if(bound < 0.0):
            bound = torch.zeros(bound.shape)
        return self.from_torch(ref, bound)


class GapBoundLever(Module):

    def forward(self, alpha, m, delta):

        bound = alpha*math.sqrt((1.0/(2.0*m))*math.log(6.0*math.sqrt(m)/delta))
        bound = bound + alpha**2.0/(8*m)
        bound = bound + math.log(6.0*math.sqrt(m)/delta)
        bound = (1.0/m)*bound

        return bound


class GapBoundDziugaite(Module):

    def __init__(self):
        super().__init__()
        self._bound_comp = Module("GapBoundComp")

    def forward(
        self, comp_prior_prior, comp_prior_post,
        comp_post_prior, comp_post_post, alpha_prior, m, delta
    ):
        bound = (self._bound_comp(
            comp_prior_prior, comp_prior_post,
            comp_post_prior, comp_post_post, m, delta)
            + 2.0*alpha_prior/m
        )
        return bound


# --------------------------------------------------------------------------- #


class EmpRiskBound(Module):

    def forward(self, risk, bound, bound_type="seeger"):
        ref = risk
        risk, bound = self.to_torch(risk, bound)

        if(bound_type == "mcallester"):
            bound = risk + np.sqrt(0.5*bound)
        elif(bound_type == "seeger"):
            bound = kl_inv(risk.item(), bound, "MAX")
        else:
            raise ValueError("bound_type must be mcallester or seeger")

        return self.from_torch(ref, bound)
