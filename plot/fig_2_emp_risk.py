import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

alpha_size = 5
seed_size = 5
data_text_dict = {
    "mnist": "MNIST",
    "fashion": "FashionMNIST",
}

for data in ["mnist", "fashion"]:

    d = NDData(
        os.path.join(
            os.path.join(os.path.dirname(__file__), ".."), "result_fig_1.csv"))

    # We initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=((0.5*8.27*3.0, 0.1*11.69*3.0)))
    fig.subplots_adjust(wspace=0.2, hspace=0.28)

    bound_list = ["dziugaite", "lever", "comp", "comp_prior"]
    color_bound_dict = {
        "lever": BLUE,
        "dziugaite": ORANGE,
        "comp": RED,
        "comp_prior": BLACK,
    }
    label_bound_dict = {
        "lever": r"Equation (8)",
        "dziugaite": r"Equation (9)",
        "comp": r"Corollary 4",
        "comp_prior": r"Corollary 5",
    }

    for bound in bound_list:

        # We get the bound
        if(bound in ["lever", "dziugaite", "comp"]):
            X = d.get("alpha_post", "alpha_prior", "risk_post_test", "seeger",
                      data=data, bound=bound, prior="0.0")
        elif(bound == "comp_prior"):
            X = d.get("alpha_post", "alpha_prior", "risk_post_test", "seeger",
                      data=data, bound="comp", prior="0.5")
        else:
            raise ValueError("The bound {} does not exist".format(bound))
        X["alpha_post"] = pd.to_numeric(X["alpha_post"])
        X["alpha_prior"] = pd.to_numeric(X["alpha_prior"])
        X = X.sort_values(by=["alpha_post", "alpha_prior"])
        X = X.to_numpy()

        # We get the minimum bound over the different alpha_prior
        X = np.reshape(X, (alpha_size, -1, seed_size, X.shape[1]))
        indices = np.argmin(X[:, :, :, 3], axis=1)
        i, j = np.indices((alpha_size, seed_size))
        X = X[i, indices, j, :]

        # We average over the seeds
        X_mean = np.mean(X, axis=1)
        X_std = np.std(X, axis=1)

        # We get the elements to plot
        alpha = np.arange(X_mean.shape[0])
        risk_test_mean = X_mean[:, 2]
        risk_test_std = X_std[:, 2]
        bound_mean = X_mean[:, 3]
        bound_std = X_std[:, 3]

        ax.plot(alpha, bound_mean, "-", c=color_bound_dict[bound],
                label=label_bound_dict[bound])
        ax.fill_between(alpha, bound_mean-bound_std,
                        bound_mean+bound_std, alpha=0.08,
                        color=color_bound_dict[bound])

        ax.plot(alpha, risk_test_mean, "--", c=color_bound_dict[bound])
        ax.fill_between(alpha, risk_test_mean-risk_test_std,
                        risk_test_mean+risk_test_std, alpha=0.08,
                        color=color_bound_dict[bound])

    label_list = [""]*len(alpha)
    label_list[0] = r"$\sqrt{m}$"
    label_list[-1] = r"$m$"
    label_list[
        math.ceil(len(label_list)/2)-1] = r"Concentration parameter $\alpha$"
    ax.set_xticks(alpha, label_list)
    ax.set_xlim(0, len(alpha)-1)
    ax.set_title("{}".format(data_text_dict[data]))
    os.makedirs("figures/", exist_ok=True)
    fig.savefig(
        "figures/fig_2_emp_risk_{}.pdf".format(data),
        bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

###############################################################################
handles, labels = ax.get_legend_handles_labels()
fig, ax = plt.subplots(1, 1, figsize=(0.05, 0.05))

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(
    axis='both', which='both',
    bottom=False, top=False, left=False, right=False)
ax.set_axis_off()
legend = fig.legend(
    handles, labels, loc='lower center', frameon=False, ncol=len(labels),
    bbox_to_anchor=(-0, -10))
os.makedirs("figures/", exist_ok=True)
plt.savefig("figures/fig_2_legend.pdf", bbox_inches="tight", pad_inches=-0.1)
plt.close(fig)

###############################################################################
