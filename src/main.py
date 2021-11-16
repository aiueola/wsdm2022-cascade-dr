from typing import List, Dict, Optional, Any
from pathlib import Path
import copy
import time
import pickle
import hydra
from omegaconf import DictConfig
from multiprocessing import Pool, cpu_count

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

from obp.ope.meta_slate import SlateOffPolicyEvaluation
from obp.ope.regression_model_slate import SlateRegressionModel
from obp.ope.estimators_slate import (
    SlateStandardIPS,
    SlateIndependentIPS,
    SlateRewardInteractionIPS,
    SlateCascadeDoublyRobust,
)
from obp.dataset import (
    logistic_reward_function,
    linear_behavior_policy_logit,
    SyntheticSlateBanditDataset,
)


def generate_and_obtain_dataset(
    n_rounds: int,
    n_unique_action: int,
    len_list: int,
    dim_context: int,
    reward_type: str,
    reward_structure: str,
    decay_function: str,
    click_model: Optional[str],
    eta: float,
    behavior_policy: str,  # "linear or "uniform"
    evaluation_policy: str,  # "similar" or "dissimilar"
    is_factorizable: bool,
    epsilon: float,  # 1.0 means that evaluation policy is uniform random
    random_state: int,
):
    click_model_ = click_model if click_model is not None else "none"
    decay_function_ = decay_function if reward_structure == "decay" else "none"
    is_factorizable_ = "_factorizable" if is_factorizable else ""
    behavior_policy_function = (
        linear_behavior_policy_logit if behavior_policy == "linear" else None
    )
    path_ = Path(
        hydra.utils.get_original_cwd()
        + f"/dataset/{reward_type}_{reward_structure}_{decay_function_}_{click_model_}"
    )
    path_.mkdir(exist_ok=True, parents=True)

    path_dataset = Path(
        path_
        / f"dataset_{behavior_policy}{is_factorizable_}_{n_unique_action}_{len_list}_{dim_context}_{eta}_{random_state}.pickle"
    )
    path_bandit_feedback = Path(
        path_
        / f"bandit_feedback_{behavior_policy}{is_factorizable_}_{evaluation_policy}{epsilon}_{n_rounds}_{n_unique_action}_{len_list}_{dim_context}_{eta}_{random_state}.pickle"
    )

    if path_bandit_feedback.exists():
        with open(path_bandit_feedback, "rb") as f:
            bandit_feedback = pickle.load(f)
        return bandit_feedback

    if path_dataset.exists():
        with open(path_dataset, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = SyntheticSlateBanditDataset(
            n_unique_action=n_unique_action,
            len_list=len_list,
            dim_context=dim_context,
            reward_type=reward_type,
            reward_structure=reward_structure,
            decay_function=decay_function,
            click_model=click_model,
            eta=eta,
            behavior_policy_function=behavior_policy_function,
            is_factorizable=is_factorizable,
            base_reward_function=logistic_reward_function,
            random_state=random_state,
        )
        with open(path_dataset, "wb") as f:
            pickle.dump(dataset, f)

    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds,
        return_pscore_item_position=True,
    )
    if behavior_policy_function is None:  # uniform random
        behavior_policy_logit_ = np.ones((n_rounds, n_unique_action))
        evaluation_policy_logit_ = linear_behavior_policy_logit(
            context=bandit_feedback["context"],
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
    else:
        behavior_policy_logit_ = behavior_policy_function(
            context=bandit_feedback["context"],
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
        if evaluation_policy == "similar":
            evaluation_policy_logit_ = (
                1 - epsilon
            ) * behavior_policy_logit_ + epsilon * np.ones(behavior_policy_logit_.shape)
        else:  # "dissimilar"
            evaluation_policy_logit_ = (
                1 - epsilon
            ) * -behavior_policy_logit_ + epsilon * np.ones(
                behavior_policy_logit_.shape
            )
    (
        bandit_feedback["evaluation_policy_pscore"],
        bandit_feedback["evaluation_policy_pscore_item_position"],
        bandit_feedback["evaluation_policy_pscore_cascade"],
    ) = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=bandit_feedback["action"],
        evaluation_policy_logit_=evaluation_policy_logit_,
        return_pscore_item_position=True,
    )
    bandit_feedback[
        "ground_truth_policy_value"
    ] = dataset.calc_ground_truth_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=bandit_feedback["context"],
    )
    bandit_feedback[
        "evaluation_policy_action_dist"
    ] = dataset.calc_evaluation_policy_action_dist(
        action=bandit_feedback["action"],
        evaluation_policy_logit_=evaluation_policy_logit_,
    )
    with open(path_bandit_feedback, "wb") as f:
        pickle.dump(bandit_feedback, f)
    return bandit_feedback


def evaluate_estimators(
    n_rounds: int,
    len_list: int,
    n_unique_action: int,
    dim_context: int,
    reward_type: str,
    reward_structure: str,
    interaction_function: str,
    decay_function: str,
    click_model: Optional[str],
    eta: float,
    behavior_policy: str,
    is_factorizable: bool,
    lambda_: float,
    random_state: int,
    regression_params: Dict[str, Any],
):
    start = time.time()
    print(f"random_state={random_state} started")
    # convert configurations
    if reward_structure in ["standard", "cascade"]:
        reward_structure = f"{reward_structure}_{interaction_function}"
    evaluation_policy = "similar" if lambda_ > 0 else "dissimilar"
    epsilon = 1 - abs(lambda_)
    # estimators setting
    ips = SlateStandardIPS(len_list=len_list, estimator_name="IPS")
    iips = SlateIndependentIPS(len_list=len_list, estimator_name="IIPS")
    rips = SlateRewardInteractionIPS(len_list=len_list, estimator_name="RIPS")
    cascade_dr = SlateCascadeDoublyRobust(
        len_list=len_list,
        n_unique_action=n_unique_action,
        estimator_name="Cascade-DR",
    )
    base_regression_model = SlateRegressionModel(
        base_model=DecisionTreeRegressor(**regression_params),
        len_list=len_list,
        n_unique_action=n_unique_action,
        fitting_method="iw",
    )
    # script
    bandit_feedback = generate_and_obtain_dataset(
        n_rounds=n_rounds,
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        decay_function=decay_function,
        click_model=click_model,
        eta=eta,
        behavior_policy=behavior_policy,
        evaluation_policy=evaluation_policy,
        is_factorizable=is_factorizable,
        epsilon=epsilon,
        random_state=random_state,
    )
    ope = SlateOffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=[ips, iips, rips, cascade_dr],
        base_regression_model=base_regression_model,
        is_factorizable=is_factorizable,
    )
    # squared errors and relative estimation erros
    se_dict_ = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=bandit_feedback["ground_truth_policy_value"],
        evaluation_policy_pscore=bandit_feedback["evaluation_policy_pscore"],
        evaluation_policy_pscore_item_position=bandit_feedback[
            "evaluation_policy_pscore_item_position"
        ],
        evaluation_policy_pscore_cascade=bandit_feedback[
            "evaluation_policy_pscore_cascade"
        ],
        evaluation_policy_action_dist=bandit_feedback["evaluation_policy_action_dist"],
        metric="se",
    )
    relative_ee_dict_ = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=bandit_feedback["ground_truth_policy_value"],
        evaluation_policy_pscore=bandit_feedback["evaluation_policy_pscore"],
        evaluation_policy_pscore_item_position=bandit_feedback[
            "evaluation_policy_pscore_item_position"
        ],
        evaluation_policy_pscore_cascade=bandit_feedback[
            "evaluation_policy_pscore_cascade"
        ],
        evaluation_policy_action_dist=bandit_feedback["evaluation_policy_action_dist"],
        metric="relative-ee",
    )

    finish = time.time()
    print(f"random_state={random_state} finished", format_runtime(start, finish))

    return se_dict_, relative_ee_dict_


def process(conf: Dict[str, Any], n_random_state: int):
    # multiprocess with different random state
    p = Pool(cpu_count())
    returns = []
    for random_state in range(n_random_state):
        return_ = p.apply_async(
            wrapper_evaluate_estimators, args=((conf, random_state),)
        )
        returns.append(return_)
    p.close()
    # aggregate results and save logs
    estimators_name = ["IPS", "IIPS", "RIPS", "Cascade-DR"]
    confs = defaultdict(list)
    estimators_performance = defaultdict(lambda: defaultdict(list))
    for return_ in returns:
        se_dict_, relative_ee_dict_, conf_ = return_.get()
        for estimator_name in estimators_name:
            se_ = se_dict_[estimator_name]
            relative_ee_ = relative_ee_dict_[estimator_name]
            estimators_performance[estimator_name]["se"].append(se_)
            estimators_performance[estimator_name]["relative-ee"].append(relative_ee_)
        confs["n_rounds"].append(conf_["n_rounds"])
        confs["n_unique_action"].append(conf_["n_unique_action"])
        confs["len_list"].append(conf_["len_list"])
        confs["dim_context"].append(conf_["dim_context"])
        confs["reward_type"].append(conf_["reward_type"])
        confs["reward_structure"].append(conf_["reward_structure"])
        confs["interaction_function"].append(conf_["interaction_function"])
        confs["decay_function"].append(conf_["decay_function"])
        confs["click_model"].append(conf_["click_model"])
        confs["eta"].append(conf_["eta"])
        confs["behavior_policy"].append(conf_["behavior_policy"])
        confs["is_factorizable"].append(conf_["is_factorizable"])
        confs["lambda_"].append(conf_["lambda_"])
        confs["random_state"].append(conf_["random_state"])

    save_logs(
        confs=confs,
        estimators_performance=estimators_performance,
    )


def assert_configuration(cfg: DictConfig):
    n_random_state = cfg.setting.n_random_state
    assert isinstance(n_random_state, int) and n_random_state > 0

    is_factorizable = cfg.setting.is_factorizable
    assert isinstance(is_factorizable, bool)

    # multiple candidates
    behavior_policy = cfg.setting.behavior_policy
    if isinstance(behavior_policy, str):
        assert behavior_policy in ["linear", "uniform"]
    else:
        for behavior_policy_ in behavior_policy:
            assert behavior_policy_ in ["linear", "uniform"]

    lambda_ = cfg.setting.lambda_
    assert (
        -1 <= lambda_ < 1
        if isinstance(lambda_, float)
        else -1 <= min(lambda_) and max(lambda_) <= 1
    )


def wrapper_evaluate_estimators(args):
    conf_, random_state = args
    conf_ = copy.deepcopy(conf_)
    conf_["random_state"] = random_state
    # randomly sample configuration
    random_ = check_random_state(random_state)
    if not isinstance(conf_["n_rounds"], int):
        conf_["n_rounds"] = int(random_.choice(conf_["n_rounds"]))
    conf_["len_list"] = int(random_.choice(conf_["len_list"]))
    conf_["reward_structure"] = random_.choice(conf_["reward_structure"])
    conf_["interaction_function"] = random_.choice(conf_["interaction_function"])
    conf_["decay_function"] = random_.choice(conf_["decay_function"])
    conf_["eta"] = float(random_.choice(conf_["eta"]))
    conf_["lambda_"] = float(random_.choice(conf_["lambda_"]))

    se_dict_, relative_ee_dict_ = evaluate_estimators(**conf_)
    return se_dict_, relative_ee_dict_, conf_


def save_logs(
    confs: Dict[str, List[Any]],
    estimators_performance: Dict[str, List[float]],
):
    # squared error
    df_se = pd.DataFrame()
    df_se["IPS"] = estimators_performance["IPS"]["se"]
    df_se["IIPS"] = estimators_performance["IIPS"]["se"]
    df_se["RIPS"] = estimators_performance["RIPS"]["se"]
    df_se["Cascade-DR"] = estimators_performance["Cascade-DR"]["se"]
    df_se.to_csv("squared_error.csv", index=False)
    print("squared error")
    print(df_se.describe())
    # relative-ee
    df_relative_ee = pd.DataFrame()
    df_relative_ee["IPS"] = estimators_performance["IPS"]["relative-ee"]
    df_relative_ee["IIPS"] = estimators_performance["IIPS"]["relative-ee"]
    df_relative_ee["RIPS"] = estimators_performance["RIPS"]["relative-ee"]
    df_relative_ee["Cascade-DR"] = estimators_performance["Cascade-DR"]["relative-ee"]
    df_relative_ee.to_csv("relative_ee.csv", index=False)
    print("relative-ee")
    print(df_relative_ee.describe())
    # configuration
    df_conf = pd.DataFrame()
    df_conf["n_rounds"] = confs["n_rounds"]
    df_conf["n_unique_action"] = confs["n_unique_action"]
    df_conf["len_list"] = confs["len_list"]
    df_conf["dim_context"] = confs["dim_context"]
    df_conf["reward_type"] = confs["reward_type"]
    df_conf["reward_structure"] = confs["reward_structure"]
    df_conf["interaction_function"] = confs["interaction_function"]
    df_conf["decay_function"] = confs["decay_function"]
    df_conf["click_model"] = confs["click_model"]
    df_conf["eta"] = confs["eta"]
    df_conf["behavior_policy"] = confs["behavior_policy"]
    df_conf["is_factorizable"] = confs["is_factorizable"]
    df_conf["lambda_"] = confs["lambda_"]
    df_conf["random_state"] = confs["random_state"]
    df_conf.to_csv("configuration.csv", index=False)


def format_runtime(start: int, finish: int):
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime // 60) % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    # configurations
    assert_configuration(cfg)
    conf = {
        "n_rounds": cfg.setting.n_rounds,  #
        "n_unique_action": cfg.setting.n_unique_action,
        "len_list": cfg.setting.len_list,  #
        "dim_context": cfg.setting.dim_context,
        "reward_type": cfg.setting.reward_type,
        "reward_structure": cfg.setting.reward_structure,  #
        "interaction_function": cfg.setting.interaction_function,  #
        "decay_function": cfg.setting.decay_function,  #
        "click_model": cfg.setting.click_model,
        "eta": cfg.setting.eta,  #
        "behavior_policy": cfg.setting.behavior_policy,
        "is_factorizable": cfg.setting.is_factorizable,
        "lambda_": cfg.setting.lambda_,  #
        "regression_params": cfg.regression_model_hyperparams,
    }
    n_random_state = cfg.setting.n_random_state
    # convert type
    conf["click_model"] = conf["click_model"] if conf["click_model"] != "None" else None
    # script
    process(
        conf=conf,
        n_random_state=n_random_state,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
