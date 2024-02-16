import sys
import os
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
            os.path.join(os.path.dirname(__file__), ".."),
            "result_fig_2.csv"))

    comp_list = [
        "risk_dist_l2", "risk_dist_fro",
        "risk_gap", "risk_param_norm",
        "risk_path_norm", "risk_sum_fro"
    ]
    color_comp_dict = {
        "risk_dist_l2": BLUE,
        "risk_dist_fro": ORANGE,
        "risk_param_norm": RED,
        "risk_path_norm": GREEN,
        "risk_sum_fro": MAGENTA,
        "risk_gap": BLACK,
    }
    label_comp_dict = {
        "risk_dist_l2": r"$\riskdistltwo_{\beta}$",
        "risk_dist_fro": r"$\riskdistfro_{\beta}$",
        "risk_param_norm": r"$\riskparamnorm_{\beta}$",
        "risk_path_norm": r"$\riskpathnorm_{\beta}$",
        "risk_sum_fro": r"$\risksumfro_{\beta}$",
        "risk_gap": r"$\riskgap_{\beta}$",
    }

    alpha_post_list = [
        "173.20508075688775", "628.3487915845371",
        "2279.5070569547775", "8269.535156745107"]

    for alpha_num, alpha_post in enumerate(alpha_post_list):

        # We initialize the plot
        fig, ax = plt.subplots(1, 1, figsize=((0.5*8.27*3.0, 0.1*11.69*3.0)))
        fig.subplots_adjust(wspace=0.2, hspace=0.28)

        for comp in comp_list:

            # We get the bound
            X = d.get("beta_post", "alpha_prior",
                      "risk_post_test", "seeger", data=data, prior="0.5",
                      comp_prior=comp, alpha_post=alpha_post)

            X["beta_post"] = pd.to_numeric(X["beta_post"])
            X["alpha_prior"] = pd.to_numeric(X["alpha_prior"])
            X = X.sort_values(by=["beta_post", "alpha_prior"])
            X = X.to_numpy()

            beta_size = len(np.unique(X[:, 0]))
            alpha_size = len(np.unique(X[:, 1]))

            X = np.reshape(X, (beta_size, alpha_size, -1, X.shape[1]))

            # We get the minimum bound over the different alpha_prior
            indices = np.argmin(X[:, :, :, 3], axis=1)
            i, j = np.indices((beta_size, X.shape[2]))
            X = X[i, indices, j, :]

            # We average over the seeds
            X_mean = np.mean(X, axis=1)
            X_std = np.std(X, axis=1)

            # We get the elements to plot
            beta = X_mean[:, 0]
            risk_test_mean = X_mean[:, 2]
            risk_test_std = X_std[:, 2]
            bound_mean = X_mean[:, 3]
            bound_std = X_std[:, 3]

            ax.plot(beta, bound_mean, "-", c=color_comp_dict[comp],
                    label=label_comp_dict[comp])
            ax.fill_between(beta, bound_mean-bound_std,
                            bound_mean+bound_std, alpha=0.2,
                            color=color_comp_dict[comp])

            ax.plot(beta, risk_test_mean, "--", c=color_comp_dict[comp])
            ax.fill_between(beta, risk_test_mean-risk_test_std,
                            risk_test_mean+risk_test_std, alpha=0.2,
                            color=color_comp_dict[comp])

        ax.set_xlim(beta[0], beta[-1])
        ax.set_title(r"{} (Corollary 5 with $\alpha={}$)".format(
            data_text_dict[data], int(float(alpha_post))))
        ax.set_xlabel(r"Trade-off parameter $\beta$")
        os.makedirs("figures/", exist_ok=True)
        fig.savefig(
            "figures/fig_5_reg_risk_{}_prior_{}.pdf".format(data, alpha_num),
            bbox_inches="tight")
        plt.close(fig)

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
            os.path.join(os.path.dirname(__file__), ".."),
            "result_fig_2.csv"))

    comp_list = [
        "risk_dist_l2", "risk_dist_fro",
        "risk_gap", "risk_param_norm",
        "risk_path_norm", "risk_sum_fro"
    ]
    color_comp_dict = {
        "risk_dist_l2": BLUE,
        "risk_dist_fro": ORANGE,
        "risk_param_norm": RED,
        "risk_path_norm": GREEN,
        "risk_sum_fro": MAGENTA,
        "risk_gap": BLACK,
    }
    label_comp_dict = {
        "risk_dist_l2": r"$\riskdistltwo_{\beta}$",
        "risk_dist_fro": r"$\riskdistfro_{\beta}$",
        "risk_param_norm": r"$\riskparamnorm_{\beta}$",
        "risk_path_norm": r"$\riskpathnorm_{\beta}$",
        "risk_sum_fro": r"$\risksumfro_{\beta}$",
        "risk_gap": r"$\riskgap_{\beta}$",
    }

    alpha_post_list = [
        "244.94897427831785", "969.0463085136714",
        "3833.6586254776325", "15166.394348316875"]

    for alpha_num, alpha_post in enumerate(alpha_post_list):

        # We initialize the plot
        fig, ax = plt.subplots(1, 1, figsize=((0.5*8.27*3.0, 0.1*11.69*3.0)))
        fig.subplots_adjust(wspace=0.2, hspace=0.28)

        for comp in comp_list:

            # We get the bound
            X = d.get("beta_post", "alpha_prior",
                      "risk_post_test", "seeger", data=data, prior="0.0",
                      comp_prior=comp, alpha_post=alpha_post)

            X["beta_post"] = pd.to_numeric(X["beta_post"])
            X["alpha_prior"] = pd.to_numeric(X["alpha_prior"])
            X = X.sort_values(by=["beta_post", "alpha_prior"])
            X = X.to_numpy()

            beta_size = len(np.unique(X[:, 0]))
            alpha_size = len(np.unique(X[:, 1]))

            X = np.reshape(X, (beta_size, alpha_size, -1, X.shape[1]))

            # We get the minimum bound over the different alpha_prior
            indices = np.argmin(X[:, :, :, 3], axis=1)
            i, j = np.indices((beta_size, X.shape[2]))
            X = X[i, indices, j, :]

            # We average over the seeds
            X_mean = np.mean(X, axis=1)
            X_std = np.std(X, axis=1)

            # We get the elements to plot
            beta = X_mean[:, 0]
            risk_test_mean = X_mean[:, 2]
            risk_test_std = X_std[:, 2]
            bound_mean = X_mean[:, 3]
            bound_std = X_std[:, 3]

            ax.plot(beta, bound_mean, "-", c=color_comp_dict[comp],
                    label=label_comp_dict[comp])
            ax.fill_between(beta, bound_mean-bound_std,
                            bound_mean+bound_std, alpha=0.2,
                            color=color_comp_dict[comp])

            ax.plot(beta, risk_test_mean, "--", c=color_comp_dict[comp])
            ax.fill_between(beta, risk_test_mean-risk_test_std,
                            risk_test_mean+risk_test_std, alpha=0.2,
                            color=color_comp_dict[comp])

        ax.set_xlim(beta[0], beta[-1])
        ax.set_title(r"{} (Corollary 4 with $\alpha={}$)".format(
            data_text_dict[data], int(float(alpha_post))))
        ax.set_xlabel(r"Trade-off parameter $\beta$")
        os.makedirs("figures/", exist_ok=True)
        fig.savefig(
            "figures/fig_5_reg_risk_{}_{}.pdf".format(data, alpha_num),
            bbox_inches="tight")
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
plt.axis('off')
os.makedirs("figures/", exist_ok=True)
plt.savefig("figures/fig_5_legend.pdf", bbox_inches="tight", pad_inches=-0.1)
plt.close(fig)


###############################################################################
