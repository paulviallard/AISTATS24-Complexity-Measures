import argparse
import copy
from h5py import File
import logging
import numpy as np
import os
import random
import torch

from module.module import Module
from core.nd_data import NDData

from learner.complexity_learner import ComplexityLearner
from learner.predict_learner import PredictLearner
from module.model import Model
from core.writer import Writer

###############################################################################


def main():
    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    arg_parser = argparse.ArgumentParser(description='')

    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="path csv")
    arg_parser.add_argument(
        "--seed", metavar="seed", default=0, type=int,
        help="seed")

    # SGLD
    arg_parser.add_argument(
        "--batch_size", metavar="batch_size", default=1, type=int,
        help="batch_size")
    arg_parser.add_argument(
        "--prior", metavar="prior", default=0.0, type=float,
        help="prior")
    arg_parser.add_argument(
        "--decay", metavar="decay", default=0.0, type=float,
        help="decay")
    arg_parser.add_argument(
        "--epoch", metavar="epoch", default=1, type=int,
        help="epoch")

    # Complexity measure
    arg_parser.add_argument(
        "--neural_path", metavar="neural_path", default=None, type=str,
        help="")
    arg_parser.add_argument(
        "--comp_prior", metavar="comp_prior", default="", type=str,
        help="")
    arg_parser.add_argument(
        "--comp_post", metavar="comp_post", default="", type=str,
        help="")
    arg_parser.add_argument(
        "--alpha_prior", metavar="alpha_prior", default=1.0, type=float,
        help="")
    arg_parser.add_argument(
        "--alpha_post", metavar="alpha_post", default=1.0, type=float,
        help="")
    arg_parser.add_argument(
        "--beta_prior", metavar="beta_prior", default=0.5, type=float,
        help="")
    arg_parser.add_argument(
        "--beta_post", metavar="beta_post", default=0.5, type=float,
        help="")

    # Bound
    arg_parser.add_argument(
        "--bound", metavar="bound", default="", type=str,
        help="")
    arg_parser.add_argument(
        "--delta", metavar="delta", default=0.05, type=float,
        help="")

    # ----------------------------------------------------------------------- #

    arg_list = arg_parser.parse_known_args()[0]

    data = arg_list.data
    path = arg_list.path
    seed = arg_list.seed*10

    # SGLD
    batch_size = arg_list.batch_size
    prior = arg_list.prior
    decay = arg_list.decay
    epoch = arg_list.epoch

    # Complexity measure
    neural_path = None
    if(arg_list.neural_path is not None):
        neural_path = os.path.join(
            os.path.dirname(__file__), arg_list.neural_path)
    comp_prior_type = arg_list.comp_prior
    comp_post_type = arg_list.comp_post
    alpha_prior = arg_list.alpha_prior
    alpha_post = arg_list.alpha_post
    beta_prior = arg_list.beta_prior
    beta_post = arg_list.beta_post

    # Bound
    bound_type = arg_list.bound
    delta = arg_list.delta

    # ----------------------------------------------------------------------- #

    device_list = ["cuda", "cpu"]

    # ----------------------------------------------------------------------- #

    data = File(os.path.join("data", data+".h5"), "r")

    x_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    x_mean, x_std = Module("MeanStd")(x_train)
    x_train = Module("StandardScaler")(x_train, x_mean=x_mean, x_std=x_std)
    y_train = np.expand_dims(y_train, 1)
    y_train = y_train.astype(np.int64)

    if(bound_type == "comp"):

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        permutation = np.arange(x_train.shape[0])
        np.random.shuffle(permutation)
        x_train = x_train[permutation]
        y_train = y_train[permutation]

        m = len(x_train)-int(prior*len(x_train))

        x_post = x_train[:m]
        y_post = y_train[:m]
        x_prior = x_train[m:]
        y_prior = y_train[m:]
    elif(bound_type == "dziugaite"):
        x_post = x_train
        y_post = y_train
        x_prior = x_train
        y_prior = y_train
    elif(bound_type == "lever"):
        x_post = x_train
        y_post = y_train
        x_prior = x_train
        y_prior = y_train
    else:
        raise RuntimeError(
            "The bound must be either comp, dziugaite, or lever")

    m = len(x_post)
    input_size = list(x_post.shape[1:])
    x_test = np.array(data["x_test"])
    x_test = Module("StandardScaler")(x_test, x_mean=x_mean, x_std=x_std)
    y_test = np.array(data["y_test"])
    y_test = np.expand_dims(y_test, 1)

    # ----------------------------------------------------------------------- #
    # Initialization of the neural complexity measure (when needed)
    if(neural_path is not None):
        f_ = Writer(neural_path, mode="h5")
        f_.load()
        neural_state_dict = f_.file_dict
        for key in neural_state_dict:
            neural_state_dict[key] = torch.tensor(neural_state_dict[key])[0]
        del f_
        model_neural = Model("MNISTComplexityMeasure")
        model_neural.load_state_dict(neural_state_dict)
        model_neural.to(device_list)
    else:
        model_neural = None

    # ----------------------------------------------------------------------- #
    # Intialization of the model
    model_init = Model("MNISTModel", seed=seed+1, input_size=input_size)
    model_init.to(device_list)

    # ----------------------------------------------------------------------- #
    # We sample the prior
    if(bound_type in ["comp", "dziugaite"]):
        logging.info("Sampling the prior...\n")

        model_prior = Model("MNISTModel", seed=seed+2, input_size=input_size)
        model_prior.to(device_list)

        if((bound_type == "comp" and prior > 0.0)
           or (bound_type == "dziugaite")):

            if(comp_prior_type in
               ["dist_l2", "sum_fro", "dist_fro", "param_norm",
                "path_norm", "gap", "neural", "dist_neural",
                "dist_dist_l2", "dist_sum_fro", "dist_dist_fro",
                "dist_param_norm", "dist_path_norm", "dist_gap", "dist_l2"]
               ):
                raise ValueError(
                    "comp_prior is conflicting")

            comp_prior = Module(
                "Complexity", comp_prior_type, 1.0, beta_prior)
            learner_post = ComplexityLearner(
                model_init, model_prior, comp_prior, alpha_prior, decay,
                batch_size, epoch, seed_list=[seed+4, seed+5])
            learner_post.fit(x_prior, y_prior, x_test, y_test)
        else:
            comp_prior = Module("Complexity", "none")

    # ----------------------------------------------------------------------- #
    # We sample the posterior
    logging.info("Sampling the posterior...\n")

    model_post = Model("MNISTModel", seed=seed+3, input_size=input_size)
    model_post.to(device_list)

    if(comp_post_type in
       ["dist_l2", "sum_fro", "dist_fro", "param_norm",
        "path_norm", "gap", "neural", "dist_neural",
        "dist_dist_l2", "dist_sum_fro", "dist_dist_fro",
        "dist_param_norm", "dist_path_norm", "dist_gap", "dist_l2"]
       ):
        learner_sgd = ComplexityLearner(
            model_init, model_post, None, None,
            decay, batch_size, epoch, seed_list=[seed+8, seed+9])
        learner_sgd.fit(x_post, y_post, x_test, y_test)
        model_sgd = copy.deepcopy(model_post)
    else:
        model_sgd = None
    comp_post = Module(
        "Complexity", comp_post_type, 1.0, beta_post, model_neural, model_sgd)
    learner_post = ComplexityLearner(
        model_init, model_post, comp_post, alpha_post,
        decay, batch_size, epoch, seed_list=[seed+6, seed+7])
    learner_post.fit(x_post, y_post, x_test, y_test)

    # ----------------------------------------------------------------------- #
    # We predict the examples in the training and the test set

    if(bound_type in ["dziugaite", "comp"]):
        pred_prior = PredictLearner(model_prior, batch_size)
        yp_prior_post = pred_prior.fit({"x_train_batch": x_post})
        yp_prior_prior = pred_prior.fit({"x_train_batch": x_prior})
        yp_prior_test = pred_prior.fit({"x_train_batch": x_test})

    pred_post = PredictLearner(model_post, batch_size)
    yp_post_post = pred_post.fit({"x_train_batch": x_post})
    yp_post_prior = pred_post.fit({"x_train_batch": x_prior})
    yp_post_test = pred_post.fit({"x_train_batch": x_test})

    if(model_sgd is not None):
        pred_sgd = PredictLearner(model_sgd, batch_size)
        yp_sgd_post = pred_sgd.fit({"x_train_batch": x_post})
        yp_sgd_prior = pred_sgd.fit({"x_train_batch": x_prior})
        yp_sgd_test = pred_sgd.fit({"x_train_batch": x_test})

    # ----------------------------------------------------------------------- #
    # We compute the complexity

    if(bound_type in ["dziugaite", "comp"]):
        logging.info("Computing the complexities...\n")

        comp_prior = Module(
            "Complexity", comp_prior_type, alpha_prior, beta_prior)
        comp_post = Module(
            "Complexity", comp_post_type,
            alpha_post, beta_post, model_neural, model_sgd)

        if(model_sgd is not None):
            comp_sgd_prior = comp_prior(
                model_init, model_sgd,
                yp_sgd_prior, y_prior, yp_sgd_test, y_test)
            comp_sgd_post = comp_post(
                model_init, model_sgd,
                yp_sgd_post, y_post, yp_sgd_test, y_test)

        # comp_model_data
        comp_prior_post = comp_post(
            model_init, model_prior,
            yp_prior_post, y_post, yp_prior_test, y_test)
        comp_prior_prior = comp_prior(
            model_init, model_prior,
            yp_prior_prior, y_prior, yp_prior_test, y_test)
        comp_post_post = comp_post(
            model_init, model_post,
            yp_post_post, y_post, yp_post_test, y_test)
        comp_post_prior = comp_prior(
            model_init, model_post,
            yp_post_prior, y_prior, yp_post_test, y_test)
    else:
        comp_prior_post = -1
        comp_prior_prior = -1
        comp_post_post = -1
        comp_post_prior = -1

    # $(\mu(h', {\cal S})-\omega(h')) - (\mu(h, {\cal S})-\omega(h))$
    if(bound_type == "comp"):
        comp = ((comp_prior_post-comp_prior_prior)
                - (comp_post_post-comp_post_prior))
    else:
        comp = -1

    # ----------------------------------------------------------------------- #
    # We compute the bounds on the gap
    logging.info("Computing the bounds on the gap...\n")

    if(bound_type == "comp"):
        bound = Module("GapBoundComp")(
            comp_prior_prior, comp_prior_post,
            comp_post_prior, comp_post_post,
            m, delta)
    elif(bound_type == "lever"):
        bound = Module("GapBoundLever")(alpha_post, m, delta)
    elif(bound_type == "dziugaite"):
        bound = Module("GapBoundDziugaite")(
            comp_prior_prior, comp_prior_post,
            comp_post_prior, comp_post_post, alpha_prior, m, delta)
    else:
        raise RuntimeError(
            "The bound must be either dziugaite, lever or comp ")

    # ----------------------------------------------------------------------- #
    # We compute the risks
    logging.info("Computing the risks...\n")

    if(bound_type == "dziugaite" or (bound_type == "comp" and prior > 0.0)):
        risk_prior_prior = Module("ZeroOne")(yp_prior_prior, y_prior)
        risk_prior_post = Module("ZeroOne")(yp_prior_post, y_post)
        risk_prior_test = Module("ZeroOne")(yp_prior_test, y_test)
        risk_post_prior = Module("ZeroOne")(yp_post_prior, y_prior)
    else:
        risk_prior_prior = -1
        risk_prior_post = -1
        risk_prior_test = -1
        risk_post_prior = -1

    if(bound_type == "comp" and model_sgd is not None):
        risk_sgd_post = Module("ZeroOne")(yp_sgd_post, y_post)
        risk_sgd_test = Module("ZeroOne")(yp_sgd_test, y_test)
        if(prior > 0.0):
            risk_sgd_prior = Module("ZeroOne")(yp_sgd_prior, y_prior)
        else:
            risk_sgd_prior = -1

    risk_post_post = Module("ZeroOne")(yp_post_post, y_post)
    risk_post_test = Module("ZeroOne")(yp_post_test, y_test)

    # ----------------------------------------------------------------------- #
    # We compute the bounds on the test risk
    logging.info("Computing the bounds on the test risk...\n")

    if(bound_type == "comp"):
        mcallester = Module("EmpRiskBound")(
            risk_post_post, bound, "mcallester")
        seeger = Module("EmpRiskBound")(
            risk_post_post, bound, "seeger")
    elif(bound_type == "lever"):
        mcallester = Module("EmpRiskBound")(
            risk_post_post, bound, "mcallester")
        seeger = Module("EmpRiskBound")(
            risk_post_post, bound, "seeger")
    elif(bound_type == "dziugaite"):
        mcallester = Module("EmpRiskBound")(
            risk_post_post, bound, "mcallester")
        seeger = Module("EmpRiskBound")(
            risk_post_post, bound, "seeger")
    else:
        raise RuntimeError(
            "The bound must be either dziugaite, lever or comp ")

    # ----------------------------------------------------------------------- #
    # Saving

    save_dict = {
        # Complexity
        "comp": comp,
        "comp_prior_prior": comp_prior_prior,
        "comp_prior_post": comp_prior_post,
        "comp_post_prior": comp_post_prior,
        "comp_post_post": comp_post_post,
        # Risks
        "risk_prior_prior": risk_prior_prior,
        "risk_prior_post": risk_prior_post,
        "risk_prior_test": risk_prior_test,
        "risk_post_prior": risk_post_prior,
        "risk_post_post": risk_post_post,
        "risk_post_test": risk_post_test,
        # Generalization bound
        "bound": bound,
        "mcallester": mcallester,
        "seeger": seeger,
    }
    if(model_sgd is not None):
        save_dict.update({
            "comp_sgd_post": comp_sgd_post,
            "comp_sgd_prior": comp_sgd_prior,
            "risk_sgd_post": risk_sgd_post,
            "risk_sgd_test": risk_sgd_test,
            "risk_sgd_prior": risk_sgd_prior,
        })

    dump = {
        "data": arg_list.data,
        "seed": seed,
        # SGLD
        "batch_size": batch_size,
        "prior": prior,
        "decay": decay,
        "epoch": epoch,
        # Complexity
        "comp_prior": comp_prior_type,
        "comp_post": comp_post_type,
        "alpha_prior": alpha_prior,
        "alpha_post": alpha_post,
        "beta_prior": beta_prior,
        "beta_post": beta_post,
        # Generalization bound
        "bound": bound_type,
        "delta": delta,
        }
    if(neural_path is not None):
        dump.update({
            "neural_path": os.path.split(neural_path)[1]
        })
    save_data = NDData(os.path.join(os.path.dirname(__file__), path))
    save_data.set(save_dict, dump)

###############################################################################


if __name__ == "__main__":
    main()
