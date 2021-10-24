import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")


df = pd.read_csv("./logs/default/squared_error.csv")
df_conf = pd.read_csv("./logs/default/configuration.csv")
df = pd.concat([df, df_conf], axis=1)

estimators = ["IPS", "IIPS", "RIPS", "Cascade-DR"]
configurations = list(df_conf.columns)

# aggregate results by "len_list"
for reward_structure in ["standard", "cascade", "independent"]:
    estimator_names, len_lists, mses = [], [], []
    df_ = df.query(f"reward_structure == '{reward_structure}'")

    for estimator in estimators:
        for len_list in range(3, 8):
            estimator_names.append(estimator)
            len_lists.append(len_list)
            mses.append(
                df_[df_["len_list"] == len_list][estimator].mean()
                / df_[df_["len_list"] == len_list]["Cascade-DR"].mean()
            )

    sns.lineplot(
        x=len_lists,
        y=mses,
        hue=estimator_names,
        style=estimator_names,
        markers=True,
        dashes=False,
        markersize=10,
    )
    plt.xlabel("slate size (L)")
    plt.ylabel("MSE (relative)")
    plt.xticks(np.arange(3, 8))
    plt.legend(loc="upper right")
    plt.savefig(
        f"./figs/slate size ({reward_structure}).png", dpi=300, bbox_inches="tight"
    )
    plt.close()

# aggregate results by "lambda_",
# which corresponds to the policy similarity between the behavior and evaluation policies
df["similarity"] = df["lambda_"]

for reward_structure in ["standard", "cascade", "independent"]:
    estimator_names, similarities, mses = [], [], []
    df_ = df.query(f"reward_structure == '{reward_structure}'")

    for estimator in estimators:
        for similarity in [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]:
            estimator_names.append(estimator)
            similarities.append(similarity)
            mses.append(
                df_[np.isclose(df_["similarity"], similarity)][estimator].mean()
                / df_[df_["similarity"] == similarity]["Cascade-DR"].mean()
            )

    sns.lineplot(
        x=similarities,
        y=mses,
        hue=estimator_names,
        style=estimator_names,
        markers=True,
        dashes=False,
        markersize=10,
    )
    plt.xlabel("evaluation policy similarity (λ)")
    plt.ylabel("MSE (relative)")
    plt.legend(loc="upper right")
    plt.savefig(
        f"./figs/evaluation policy similarity ({reward_structure}).png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

df_n = pd.read_csv("./logs/n_rounds/squared_error.csv")
df_conf_n = pd.read_csv("./logs/n_rounds/configuration.csv")
df_n = pd.concat([df_n, df_conf_n], axis=1)

# aggregate results by "n_rounds"
for reward_structure in ["standard", "cascade", "independent"]:
    estimator_names, n_rounds, mses = [], [], []
    df_ = df_n.query(f"reward_structure == '{reward_structure}'")

    for estimator in estimators:
        for n_round in [250, 500, 1000, 2000, 4000]:
            estimator_names.append(estimator)
            n_rounds.append(str(n_round))
            mses.append(
                df_[df_["n_rounds"] == n_round][estimator].mean()
                / df_[df_["n_rounds"] == n_round]["Cascade-DR"].mean()
            )
    sns.lineplot(
        x=n_rounds,
        y=mses,
        hue=estimator_names,
        style=estimator_names,
        markers=True,
        dashes=False,
        markersize=10,
    )

    plt.xlabel("data size (n)")
    plt.ylabel("MSE (relative)")
    plt.xticks(["250", "500", "1000", "2000", "4000"])
    plt.legend(loc="upper right")
    plt.savefig(
        f"./figs/data size ({reward_structure}).png", dpi=300, bbox_inches="tight"
    )
    plt.close()
