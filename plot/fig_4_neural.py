import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.nd_data import NDData

###############################################################################

path = os.path.dirname(__file__)
path = os.path.abspath(path)+os.sep
if(os.name == 'nt'):
    path = path.replace("\\", "/")

f = open(os.path.join(os.path.dirname(__file__), "header_standalone.tex"), "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)
plt.rcParams.update({
    "font.size": 20,
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    "text.latex.preamble": preamble,
    "pgf.preamble": preamble,
})

###############################################################################

WHITE = "#FFFFFF"
BLACK = "#000000"
BLUE = "#0077BB"
CYAN = "#009988"
GREEN = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
GREY = "#BBBBBB"

###############################################################################

seed_size = 5
data_text_dict = {
    "mnist": "MNIST",
    "fashion": "FashionMNIST",
}

for data in ["mnist", "fashion"]:

    d = NDData(
        os.path.join(
            os.path.join(os.path.dirname(__file__), ".."),
            "result_fig_3.csv"))

    # We initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=((0.5*8.27*5.0, 0.1*11.69*5.0)))
    fig.subplots_adjust(wspace=0.2, hspace=0.28)

    comp_list = [
        "dist_dist_l2", "dist_sum_fro", "dist_dist_fro",
        "dist_param_norm", "dist_path_norm", "dist_neural", "dist_gap"]
    label_comp_dict = {
        "dist_dist_l2": r"$\distdistltwo$",
        "dist_dist_fro": r"$\distdistfro$",
        "dist_param_norm": r"$\distparamnorm$",
        "dist_path_norm": r"$\distpathnorm$",
        "dist_sum_fro": r"$\distsumfro$",
        "dist_gap": r"$\distgap$",
        "dist_neural": r"$\distneural$",
    }
    label_list = []

    i = 0
    for comp in comp_list:

        label_list.append(label_comp_dict[comp])

        if(comp not in ["dist_neural", "neural"]):
            # We get the bound
            X = d.get("risk_post_test", "seeger",
                      data=data, prior="0.0", comp_post=comp,
                      alpha_post="60000.0")

            X = X.to_numpy()
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
        else:
            X = d.get("risk_post_test", "seeger", "neural_path",
                      data=data, prior="0.0", comp_post=comp,
                      alpha_post="60000.0")

            X = X.sort_values(by=["neural_path"])
            X.to_numpy()
            X = np.reshape(X, (-1, seed_size, 3))
            X_mean = np.mean(X[:, :, :2], axis=1)
            X_std = np.std(X[:, :, :2].astype(float), axis=1)
            indices = np.argmin(X_mean[:, 1], axis=0)
            X_mean = X_mean[indices, :]
            X_std = X_std[indices, :]

        ax.bar(i, X_mean[1], yerr=X_std[1],
               alpha=1.0, hatch="/", color=WHITE, edgecolor=BLACK,
               error_kw=dict(lw=3, ecolor=BLACK, capsize=10, capthick=1))
        ax.bar(i, X_mean[0], yerr=X_std[0],
               alpha=0.6, color=ORANGE,
               error_kw=dict(lw=3, ecolor=BLACK, capsize=10, capthick=1))

        if(comp in ["dist_neural", "neural"]):
            y_lim_max = X_mean[1]+0.1

        i += 1

    ax.set_title("{}".format(data_text_dict[data]), fontsize=35)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6], label_list, fontsize=25)
    os.makedirs("figures/", exist_ok=True)
    fig.savefig(
        "figures/fig_4_dist_{}.pdf".format(data),
        bbox_inches="tight")
    plt.close(fig)

###############################################################################

seed_size = 5
data_text_dict = {
    "mnist": "MNIST",
    "fashion": "FashionMNIST",
}

for data in ["mnist", "fashion"]:

    d = NDData(
        os.path.join(
            os.path.join(os.path.dirname(__file__), ".."),
            "result_fig_3.csv"))

    # We initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=((0.5*8.27*5.0, 0.1*11.69*5.0)))
    fig.subplots_adjust(wspace=0.2, hspace=0.28)

    comp_list = [
        "dist_l2", "sum_fro", "dist_fro", "param_norm", "path_norm",
        "neural", "gap"]
    label_comp_dict = {
        "dist_l2": r"$\distltwo$",
        "dist_fro": r"$\distfro$",
        "param_norm": r"$\paramnorm$",
        "path_norm": r"$\pathnorm$",
        "sum_fro": r"$\sumfro$",
        "gap": r"$\gap$",
        "neural": r"$\neural$",
    }
    label_list = []

    i = 0
    for comp in comp_list:

        label_list.append(label_comp_dict[comp])

        if(comp not in ["dist_neural", "neural"]):
            # We get the bound
            X = d.get("risk_post_test", "seeger",
                      data=data, prior="0.0", comp_post=comp,
                      alpha_post="60000.0")

            X = X.to_numpy()
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
        else:
            X = d.get("risk_post_test", "seeger", "neural_path",
                      data=data, prior="0.0", comp_post=comp,
                      alpha_post="60000.0")

            X = X.sort_values(by=["neural_path"])
            X.to_numpy()
            X = np.reshape(X, (-1, seed_size, 3))
            X_mean = np.mean(X[:, :, :2], axis=1)
            X_std = np.std(X[:, :, :2].astype(float), axis=1)
            indices = np.argmin(X_mean[:, 1], axis=0)
            X_mean = X_mean[indices, :]
            X_std = X_std[indices, :]

        ax.bar(i, X_mean[1], yerr=X_std[1],
               alpha=1.0, hatch="/", color=WHITE, edgecolor=BLACK,
               error_kw=dict(lw=3, ecolor=BLACK, capsize=10, capthick=1))
        ax.bar(i, X_mean[0], yerr=X_std[0],
               alpha=0.6, color=ORANGE,
               error_kw=dict(lw=3, ecolor=BLACK, capsize=10, capthick=1))

        if(comp in ["dist_neural", "neural"]):
            y_lim_max = X_mean[1]+0.1

        i += 1

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6], label_list, fontsize=25)
    os.makedirs("figures/", exist_ok=True)
    fig.savefig(
        "figures/fig_4_{}.pdf".format(data),
        bbox_inches="tight")
    plt.close(fig)
