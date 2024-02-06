from learners.utils import load_expected_post, log_no_nan, exp_int_no_nan
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy as np


def _prepare_data(data, features, logspace):
    X, y = load_expected_post(data, features)
    if logspace:
        X = log_no_nan(X)
        y = log_no_nan(y)
    return X, y


def to_string(model, features, logspace, d=0):
    coefs = model.coef_

    def round_d(num):
        if d == 0:
            return int(round(num, 0))
        else:
            return round(num, d)

    if not logspace:
        lst = [
            "({})*{}".format(round_d(coefs[c]), features[c])
            for c in range(len(coefs))
            if round_d(coefs[c]) != 0
        ]
        if len(lst) != 0:
            string = " + ".join(lst)
        else:
            string = "0"
    else:
        lst = [
            f"({features[c]})**({round_d(coefs[c])})"
            for c in range(len(coefs))
            if round_d(coefs[c]) != 0
        ]
        if len(lst) != 0:
            string = " * ".join(lst)
        else:
            string = "1"

    if logspace:
        string += "*{}".format(
            str(round_d(eval(f"exp_int_no_nan({model.intercept_})")))
        )
    else:
        string += "+{}".format(str(round_d(model.intercept_)))
    return string


def loss(y, y_pred):
    res = [(y1 - y2) ** 2 for y1, y2 in zip(y, y_pred)]
    indices = [i for i, element in enumerate(res) if element <= 0.06]
    return indices


def remove_samples(X, y, y_pred):
    idx = loss(y, y_pred)
    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx)
    return X, y


def learn_reg_models(data, features, logspace):
    models = []
    X_init, Y_post = _prepare_data(data, features, logspace)
    # reg = LinearRegression(fit_intercept=True)
    # reg.fit(X_init, Y_post)
    # y_pred = reg.predict(X_init)
    # print(f"{features=}")
    # print(reg.coef_)
    # print(reg.intercept_)
    # print(to_string(reg, features, logspace))
    while len(X_init) > 0:
        print(len(X_init))
        X, y = X_init, Y_post
        reg = LinearRegression(fit_intercept=True)
        # reg = linear_model.Lasso(alpha=0.001)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        X_init, Y_post = remove_samples(X, y, y_pred)
        if len(X_init) < len(X):
            print("model inserted")
            models.append(reg)
    for model in models:
        print(to_string(model, features, logspace))
