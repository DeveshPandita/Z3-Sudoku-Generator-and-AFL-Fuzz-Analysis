from wolframclient.evaluation import WolframLanguageSession
import pandas as pd
import os
import time
from loop import cegis_one_prog
import sys
import json
import argparse
import copy


parser = argparse.ArgumentParser()
parser.add_argument(
    "-cached",
    "--cached",
    help="put 'yes' (or 'y') \
    if you want to use cached program traces as the data; otherwise, \
        the program would sample new data",
)
parser.add_argument(
    "-nruns",
    "--nruns",
    help="number of runs from each initial\
    state",
    type=int,
)
parser.add_argument(
    "-nstates",
    "--nstates",
    help="number of initial states \
    we sample for each benchmark",
    type=int,
)
parser.add_argument("-test", "--test", help="put 'yes' (or 'y') if running smoke test")

args = parser.parse_args()

# By default, the program will sample new data
if str(args.cached).lower() in ["yes", "y"]:
    sample_new_data = False
else:
    sample_new_data = True

if str(args.test).lower() in ["yes", "y"]:
    smoke_test = True
else:
    smoke_test = False

if args.nruns is None:
    NUM_runs = 500
else:
    NUM_runs = args.nruns

if args.nstates is None:
    NUM_init_states = 1000
else:
    NUM_init_states = args.nruns


PATH = os.path.realpath("")
assumed_shape = "post + [G] * model"

"""
Given: 
    [features_log]: a list of strings denoting 
                    expressions generated for the multiplicative models 
                    (we call it `log mode` because it's implemented through 
                    taking the logarithms of the data)
    [features_linear]: a list of strings denoting 
                    expressions generated for the linear models (`linear mode`)
    [log_not_split]: a list of strings in [features_log] that denote
                    expressions involving probability variables
                    (in general, we don't want the predicates in the model trees 
                    to split on probabilities)
    [linear_not_split]: a list of strings in [features_linear] that denote
                        expressions involving probability variables
    [config]: a dictionary loaded from a .json file under `\configuration_files`
                
                [config] is not used in [prepare_tree_learner]
                but we have it so [prepare_tree_learner], [prepare_NN_sub], 
                [prepare_NN_exact] have the same type

Returns:
    [Learners]: a list of objects that take in data and output models (see class 
                [Learner] of `learners/abstract_learner.py`; we call them 
                learners). More specifically, [prepare_tree_learner(...)] 
                returns two learners of class [TreeLearner] (see 
                `learners/Tree.py`. ) They both train model trees in the 
                conventional way (using divde and conquer), with the objective 
                of finding a model tree that translates to an exact invariant. 
                One learner assumes linear models on the leaves, and 
                the another learner assumes multiplicative models on the leaves. 
"""
# """
# [get_config] loads the json object in [progname]'s configuration file
# """


def get_config(progname):
    with open(os.path.join(PATH, "configuration_files", progname + ".json"), "r") as f:
        config = json.load(f)
    return config


def config_select_pre(config):
    config_idx = copy.deepcopy(config)
    config_idx["wp"]["pre"] = config_idx["wp"]["pre"]
    return config_idx


# session = WolframLanguageSession()
with open(os.path.join(PATH, "program_list.txt"), "r") as f:
    prognames = f.read().strip().split("\n")

if smoke_test:
    prognames = ["Geo0"]

results = {}
for progname in prognames[:1]:
    config = get_config(progname)
    post = config["wp"]["post"]
    guard = config["wp"]["guard"]
    pre = config["wp"]["pre"]
    config_idx = config_select_pre(config)
    print("     Benchmark name: {}; Post-exp: {}".format(progname, post))
    cegis_one_prog(
        progname, config_idx, NUM_runs, NUM_init_states, sample_new_data, assumed_shape
    )
