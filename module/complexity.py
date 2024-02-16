import re
import torch
import numpy as np
from module.module import Module
###############################################################################


class BoundedCrossEntropy(Module):

    def __init__(self, L_max=4.0):
        super().__init__()
        self.L_max = L_max

    def forward(self, out, y):
        # Computing bounded cross entropy (from Dziugaite et al., 2018)
        ref = out
        out, y = self.to_torch(out, y)

        exp_L_max = torch.exp(-torch.tensor(self.L_max, requires_grad=False))
        out_ = exp_L_max + (1.0-2.0*exp_L_max)*out
        out_ = (1.0/self.L_max)*torch.log(out_)
        loss = torch.nn.functional.nll_loss(out_, y[:, 0])

        return self.from_torch(ref, loss)

###############################################################################


class Softmax(Module):

    def forward(self, x, dim=1):
        # Computing bounded cross entropy (from Dziugaite et al., 2018)
        ref = x
        x = self.to_torch(x)
        x = torch.softmax(x, dim=dim)
        return self.from_torch(ref, x)

###############################################################################


# Norm considered in the complexity measures
class ComplexityNorm(Module):

    def __init__(self, comp_name, alpha=1.0):
        super().__init__()
        self.__comp_name = comp_name
        self.__alpha = float(alpha)

    def get_param_list(self, model):

        param_list = list(model.parameters())
        param_list_cat = []
        for param in param_list:
            param_list_cat.append(param.view(-1))
        param_list_cat = torch.cat(param_list_cat, dim=0)
        return param_list, param_list_cat

    def get_path_norm(self, model):

        def path_linear(input, weight, bias=None):
            return old_linear(input, weight, bias)

        def path_conv2d(input, weight, bias=None,
                        stride=1, padding=0, dilation=1, groups=1):
            return old_conv2d(input, weight**2.0, bias**2.0,
                              stride, padding, dilation, groups)

        old_conv2d = torch.nn.functional.conv2d
        old_linear = torch.nn.functional.linear
        torch.nn.functional.conv2d = path_conv2d
        torch.nn.functional.linear = path_linear

        x_shape = [1] + model.input_size
        x_ = torch.ones(x_shape, device=model.device)
        y_pred = model({"x": x_})
        norm = torch.sum(y_pred)

        torch.nn.functional.conv2d = old_conv2d
        torch.nn.functional.linear = old_linear

        return norm

    def forward(self, model_init, model):
        # Based on https://github.com/nitarshan/robust-generalization-measures/
        # blob/master/data/generation/measures.py
        comp_dict = {}
        comp_name = self.__comp_name

        init_param, init_param_cat = self.get_param_list(model_init)
        param, param_cat = self.get_param_list(model)

        # ------------------------------------------------------------------- #

        #  Vector Norm Measures
        if(comp_name in ["dist_l2"]):
            comp_dict["dist_l2"] = (param_cat-init_param_cat).norm(p=2)

        # ------------------------------------------------------------------- #

        #  Frobenius Norm
        if(comp_name in ["sum_fro", "dist_fro", "param_norm"]):
            fro_norm_list = param[0].norm("fro").unsqueeze(0)
            for i in range(1, len(param)):
                fro_norm_list = torch.cat(
                    (fro_norm_list, param[i].norm("fro").unsqueeze(0)))
            d = float(len(fro_norm_list))

            dist_fro_norm_list = (
                param[0]-init_param[0]).norm("fro").unsqueeze(0)
            for i in range(1, len(param)):
                dist_fro_norm_list = torch.cat(
                    (dist_fro_norm_list,
                     (param[i]-init_param[i]).norm("fro").unsqueeze(0)))

            prod_fro = (fro_norm_list**2.0).prod()

        if(comp_name in ["sum_fro"]):
            comp_dict["sum_fro"] = (d*prod_fro**(1/d))
        if(comp_name in ["dist_fro"]):
            comp_dict["dist_fro"] = (dist_fro_norm_list**2.0).sum()
        if(comp_name in ["param_norm"]):
            comp_dict["param_norm"] = (fro_norm_list**2.0).sum()

        # ------------------------------------------------------------------- #

        #  Path Norm
        if(comp_name in ["path_norm"]):
            comp_dict["path_norm"] = self.get_path_norm(model)

        # ------------------------------------------------------------------- #

        comp = comp_dict[comp_name]
        comp = self.__alpha*comp
        return comp


class ComplexityDistNorm(Module):

    def __init__(self, comp_name, alpha, model_dist):
        super().__init__()
        self.__comp_name = comp_name
        self.__alpha = float(alpha)
        self.__model_dist = model_dist
        self.__norm = Module("ComplexityNorm", self.__comp_name, self.__alpha)
        self.__dist_init = None

    def forward(self, model_init, model):

        if(self.__dist_init is None and self.__model_dist is not None):
            self.__dist_init = self.__norm(model_init, self.__model_dist)
            self.__dist_init = float(self.__dist_init)

        comp = torch.abs(self.__norm(model_init, model)-self.__dist_init)
        return comp


# The complexity measures associated with the gap \alpha{\rm R}_{\cal S}(h)
class ComplexityRisk(Module):

    def __init__(self, alpha):
        super().__init__()
        self.__loss = Module("BoundedCrossEntropy")
        self.__alpha = alpha

    def forward(self, y_pred, y):
        ref = y_pred
        y_pred, y = self.to_torch(y_pred, y)
        y_pred = torch.softmax(y_pred, dim=1)
        comp = self.__alpha*self.__loss(y_pred, y)
        return self.from_torch(ref, comp)


# The complexity measures associated with the gap
# \alpha|{\rm R}_{\cal S}(h)-{\rm R}_{\cal T}(h)|
class ComplexityGap(Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.__risk = Module("ComplexityRisk", alpha)

    def forward(self, yp_train, y_train, yp_test, y_test):
        ref = yp_train
        yp_train, y_train, yp_test, y_test = self.to_torch(
            yp_train, y_train, yp_test, y_test)
        comp = torch.abs(
            self.__risk(yp_train, y_train)-self.__risk(yp_test, y_test))
        return self.from_torch(ref, comp)


class ComplexityDistGap(Module):

    def __init__(self, alpha, model_dist):
        super().__init__()
        self.__alpha = float(alpha)
        self.__model_dist = model_dist
        self.__gap = Module("ComplexityGap", self.__alpha)
        self.__dist_init = None

    def forward(self, yp_train, y_train, yp_test, y_test):

        if(self.__dist_init is None and self.__model_dist is not None):
            self.__dist_init = self.__gap(yp_train, y_train, yp_test, y_test)
            self.__dist_init = float(self.__dist_init)

        comp = torch.abs(
            self.__gap(yp_train, y_train, yp_test, y_test)-self.__dist_init)
        return comp


# The complexity measures written as $\alpha({\rm R}_{\cal S}(h)+\beta\|h\|)$
class ComplexityRiskNorm(Module):

    def __init__(self, comp_name, alpha=1.0, beta=0.5):
        super().__init__()
        self.__norm = Module("ComplexityNorm", comp_name, alpha)
        self.__risk = Module("ComplexityRisk", alpha)
        self.__alpha = float(alpha)
        self.__beta = float(beta)

    def forward(self, model_init, model, y_pred, y):
        ref = y_pred
        y_pred, y = self.to_torch(y_pred, y)
        comp = self.__beta*self.__risk(y_pred, y)
        comp = comp + (1.0-self.__beta)*self.__norm(model_init, model)
        return self.from_torch(ref, comp)


# The complexity measures associated with the risk+gap
# \alpha{\rm R}_{\cal S}(h) + \alpha|{\rm R}_{\cal S}(h)-{\rm R}_{\cal T}(h)|
class ComplexityRiskGap(Module):

    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.__risk = Module("ComplexityRisk", alpha)
        self.__gap = Module("ComplexityGap", alpha)
        self.__alpha = float(alpha)
        self.__beta = float(beta)

    def forward(self, yp_train, y_train, yp_test, y_test):
        ref = yp_train
        yp_train, y_train, yp_test, y_test = self.to_torch(
            yp_train, y_train, yp_test, y_test)

        comp = self.__beta*self.__risk(yp_train, y_train)
        comp = comp + (1.0-self.__beta)*self.__gap(
            yp_train, y_train, yp_test, y_test)
        return self.from_torch(ref, comp)


class ComplexityNeural(Module):

    def __init__(self, alpha, model_neural):
        super().__init__()
        self.__model_neural = model_neural
        self.__alpha = float(alpha)

    def forward(self, model):
        norm = 0.0
        for param in model.parameters():
            norm += torch.norm(param)**2.0
        norm = torch.sqrt(norm)

        batch = {}
        for i, param in enumerate(model.parameters()):
            batch["param_"+str(i)] = torch.unsqueeze(param, dim=0)
        batch["step"] = "predict"
        comp = self.__model_neural(batch)
        comp = self.__alpha*comp[0, 0]
        return comp


class ComplexityDistNeural(Module):

    def __init__(self, alpha, model_neural, model_dist):
        super().__init__()
        self.__model_neural = model_neural
        self.__model_dist = model_dist
        self.__alpha = float(alpha)

        self.__neural = None
        if(self.__model_neural is not None):
            self.__neural = Module(
                "ComplexityNeural", self.__alpha, self.__model_neural)

        self.__dist_init = None

    def forward(self, model):

        if(self.__dist_init is None and
           self.__neural is not None and self.__model_dist is not None
           ):
            self.__dist_init = self.__neural(self.__model_dist)
            self.__dist_init = float(self.__dist_init)

        comp = torch.abs(self.__neural(model)-self.__dist_init)
        return comp


# The complexity measures
class Complexity(Module):

    def __init__(
        self, comp_name, alpha=1.0, beta=0.5,
        model_neural=None, model_dist=None
    ):

        super().__init__()
        self.__risk = Module("ComplexityRisk", alpha)
        self.__risk_norm = Module(
            "ComplexityRiskNorm", comp_name.replace("risk_", ""), alpha, beta)
        self.__dist_norm = Module(
            "ComplexityDistNorm", re.sub("^dist_", "", comp_name),
            alpha, model_dist)
        self.__norm = Module("ComplexityNorm", comp_name, alpha)
        self.__risk_gap = Module("ComplexityRiskGap", alpha, beta)
        self.__gap = Module("ComplexityGap", alpha)
        self.__dist_gap = Module("ComplexityDistGap", alpha, model_dist)
        self.__neural = Module("ComplexityNeural", alpha, model_neural)
        self.__dist_neural = Module(
            "ComplexityDistNeural", alpha, model_neural, model_dist)

        self.name = comp_name
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.model_neural = model_neural

    def forward(
        self, model_init, model, yp_optim, y_optim, yp_test, y_test
    ):
        ref = yp_optim
        (yp_optim, y_optim, yp_test, y_test) = self.to_torch(
            yp_optim, y_optim, yp_test, y_test, device=model.device)

        comp_name = self.name
        if(yp_optim is None or yp_optim.shape[0] == 0):
            comp_name = "none"

        if(comp_name == "risk"):
            comp = self.__risk(yp_optim, y_optim)
        elif(comp_name in
             ["risk_dist_l2", "risk_sum_fro", "risk_dist_fro",
              "risk_param_norm", "risk_path_norm"]):
            comp = self.__risk_norm(
                model_init, model, yp_optim, y_optim)
        elif(comp_name in
             ["dist_dist_l2", "dist_sum_fro", "dist_dist_fro",
              "dist_param_norm", "dist_path_norm"]):
            comp = self.__dist_norm(
                model_init, model)
        elif(comp_name in
             ["dist_l2", "sum_fro", "dist_fro",
              "param_norm", "path_norm"]):
            comp = self.__norm(
                model_init, model)
        elif(comp_name == "risk_gap"):
            comp = self.__risk_gap(yp_optim, y_optim, yp_test, y_test)
        elif(comp_name == "gap"):
            comp = self.__gap(yp_optim, y_optim, yp_test, y_test)
        elif(comp_name == "dist_gap"):
            comp = self.__dist_gap(yp_optim, y_optim, yp_test, y_test)
        elif(comp_name == "neural"):
            comp = self.__neural(model)
        elif(comp_name == "dist_neural"):
            comp = self.__dist_neural(model)
        elif(comp_name == "none"):
            comp = torch.tensor(0.0, device=model.device)

        return self.from_torch(ref, comp)
