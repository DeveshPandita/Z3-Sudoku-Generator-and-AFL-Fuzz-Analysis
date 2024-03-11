import numpy as np
import pandas as pd
import ex_prog
import random
import ex_prog
from pathlib import Path

"""
[sample_by_type] can be seen as a helper function of [sampleStates] in Fig. 2. 
It samples one possible initial state. 

Given: 
    [var_types]: a dictionary that maps fields "Reals", "Integers", "Booleans", 
                "Probs" to a list of variables of that type
Return: 
    [dic]: a dictionary that maps each variable name appeared in [var_types] to 
            a value; the value is sampled among a subset of possible values for 
            the variable with that type, 
"""


def sample_by_type(var_types):
    dic = {}
    for thetype, vars in var_types.items():
        if thetype == "Integers":
            for var in vars:
                dic[var] = random.randint(0, 20)  # currently Hard-coded
        elif thetype == "Reals":
            for var in vars:
                dic[var] = round(random.uniform(0, 20), 2)  # currently Hard-coded
        elif thetype == "Booleans":
            for var in vars:
                dic[var] = random.randint(0, 1)
        elif thetype == "Probs":
            for var in vars:
                dic[var] = round(random.uniform(0.05, 0.95), 2)  # currently Hard-coded
        else:
            raise Exception
    return dic


"""
When [sample_new] is [True], [sample] performs two following steps we described 
in Fig. 2 and save [data] in to [filename]:
        `states ← sampleStates(feat, nstates)
         data ← sampleTraces(prog, pexp, feat, nruns, states)`
Otherwise, [sample] load [data] (which are in the form of pandas.dataframes) 
from [filename]

Given:
    [progname], [exact], [Num_runs], [Num_init] are the same as themselves in 
    the input parameters to [cegis_one_prog]. 
    [assumption]: the same as [assumed_shape] in the input parameters to 
                [cegis_one_prog]. 
    [filename]: 1 or 3 strings indicating the filename(s) to which we saves data
    [var_types]: the same as in [sample_by_type]
    [features]: a list of strings denoting features
    [task]: the dictionary returned by `get_task(config)` of cegis.py
    [sample_new]: a boolean returned by `get_sample_new_data()` of cegis.py, 
                    which indicates whether we sampling new data or load data 
                    from csv files
    [mode]: a string indicating how we write into the csv file. The default is 
            "w", which means writing over. Another choice is "a", which means 
            appending. 
    [includeheader]: a boolean indicating 
Return: 
    If [exact] is [True], we return a single pandas.dataframe. 
    When it is [False], we return one dataframe containing all initial states, 
    another dataframe containing all post states from all initial states, 
    (we separate them because we don't konw a good way to handle dataframes 
     with hierarchial structures), and a dataframe specifying the weight to 
     each data entry. The weights would allow taking a weighted average during 
     training, but currently, they are all set to 1 and not taking effect. 
"""


def sample(
    progname: str,
    assumption: str,
    filename,
    var_types: dict,
    features: list,
    task: dict,
    NUM_runs: int,
    NUM_init: int,
    sample_new: bool,
    mode="w",
    includeheader=True,
):
    if sample_new:
        data = sampler_expected_post(progname, var_types, features, NUM_runs, NUM_init)
        data = data.reindex(columns=sorted(data.columns))
        if not includeheader:
            data.to_csv(filename, mode=mode, header=None)
        else:
            data.to_csv(filename, mode=mode)
    else:
        data = pd.read_csv(filename)
        for feature in features:
            if feature not in list(data.columns):
                data[feature] = eval(feature, {"__builtins__": None}, data)
    return data


"""
Helper function for sample()
Given: 
    Everything are as in [sample(...)]
Return:
    [data]: a data frame whose columns are features plus "expected post". 
    For every initial state, we get one row of data, and its value in the column 
    "expected post" is the expected value of the postexpectation after running 
    the loop from this initial state. 
"""


def sampler_expected_post(progname, var_types, features, NUM_runs, NUM_init_states):
    traces = []
    for _ in range(NUM_init_states):
        inpt = sample_by_type(var_types)
        trace = eval(f"ex_prog.{progname}" + f"({inpt},{features},{NUM_runs})")
        if not (trace is None):
            traces.append(trace)
    return pd.DataFrame(traces)


"""
Given: 
    [counter_ex]: a list of dictionaries, where each dictionary maps program 
    variables to values
    Everything else as the same as in [sample()]
Return:
    Almost the same as [sample] except that the initial states are not randomly 
    sampled. All initical states come from [counter_ex]. 
"""


def sample_counterex(
    progname, assumption, filename, features, task, counter_ex, NUM_runs
):
    traces = [
        eval(f"ex_prog.{progname}" + f"({inpt},{features},{NUM_runs * 3})")
        for inpt in counter_ex
    ]
    # copy 30 times
    traces = [i for _ in range(30) for i in traces if not (i is None)]
    df = pd.DataFrame(traces)
    df = df.reindex(columns=sorted(df.columns))
    df.to_csv(filename, header=None, mode="a")
    print(df)
    return df


"""
    Flatten a list of list to a single list
"""


def flatten(lst):
    return [i for l in lst for i in l]


"""
    When we fill in the value of features at different states based on the 
    program variables' values on those states for [df], extra metadata is added 
    to [df] and [cleanup_df] is just to clean up that. 
"""


def cleanup_df(df):
    if "__builtins__" in df.columns:
        df = df.drop(["__builtins__"], axis=1)
    df = df.reindex(columns=sorted(df.columns))
    return df
