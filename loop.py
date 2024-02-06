import os
from reg_learn2 import learn_reg_models
import pandas as pd
from feature_generation import generate_features_linear, generate_features_log
from sampler import sample
from collections import defaultdict

PATH = os.path.realpath("")


def cegis_one_prog(
    progname: str,
    config,
    NUM_runs: int,
    NUM_init: int,
    sample_new: bool,
    assumed_shape,
):
    var_types = get_var_types(config)
    task = get_task(config)
    # The following block corresponds to `feat ← getFeatures(prog, pexp)` in Fig. 2
    add_features_dic = get_user_supplied_features(config)
    add_features_dic = defaultdict(list, add_features_dic)
    # features_log, log_not_split = generate_features_log(var_types, add_features_dic)
    features_linear, linear_not_split = generate_features_linear(
        var_types, add_features_dic
    )
    features = list(set(features_linear + linear_not_split))
    print("Exist generates the following set of features:\n     {}".format(features))
    # This block roughly corresponds to `states ← sampleStates(feat, nstates)
    #                      data ← sampleTraces(prog, pexp, feat, nruns, states)`
    # Both [sampleStates] and [sampleTraces] are implemented in [sample].
    filename = get_filename(progname)
    print("     Start sampling {}".format(progname))
    data = sample(
        progname,
        assumed_shape,
        filename,
        var_types,
        features,
        task,
        NUM_runs,
        NUM_init,
        sample_new,
    )
    models = learn_reg_models(data, features, False)
    return models


# ---------------------Basic helper functions -----------------------------------


"""
[get_var_types] returns a dictionary that maps fields "Reals", "Integers", 
"Booleans", "Prob" to a list of variables of that type
"""


def get_var_types(config):
    var_config = config["Sample_Points"]
    var_config = defaultdict(list, var_config)
    return var_config


"""
[get_task] returns [task], which given as input to [verify_cand]. 
"""


def get_task(config):
    return config["wp"]


"""
[get_user_supplied_features] finds user-supplied features if there exists any. 
It returns a dictionary that associates types ("Reals", "Probs", "Integers" and 
"Booleans") to user-supplied features of these types. 
"""


def get_user_supplied_features(config):
    try:
        add_features_dic = config["additional features for exact"]
    except KeyError:
        add_features_dic = {}
    return add_features_dic


"""
[get_filename] returns filename(s) where we store sampled data into
"""


def get_filename(progname):
    filename = os.path.join(PATH, "generated_data", progname + "_expected_post.csv")
    return filename


"""
Roughly, [combine_data] unions [data], [add_data] and [more_sample]. 
The for-loops of the second branch of [if-then-else] are there to make sure 
that [data], [add_data] and [more_sample] have the same columns
"""


def combine_data(data, add_data, more_sample):
    return pd.concat([data, add_data, more_sample], axis=0)
