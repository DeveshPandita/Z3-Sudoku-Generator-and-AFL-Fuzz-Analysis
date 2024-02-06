from learners.utils import load_expected_post, log_no_nan, exp_int_no_nan
import numpy as np
import torch
from practice2 import fit_line_or_hyperplane
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()


def to_string(w, b, features, logspace, d=0):
    print(f"this is b::{b}")

    def round_d(num):
        if d == 0:
            return int(round(num, 0))
        else:
            return round(num, d)

    if not logspace:
        lst = [
            "({})*{}".format(round_d(w[c]), features[c])
            for c in range(len(w))
            if round_d(w[c]) != 0
        ]
        if len(lst) != 0:
            string = " + ".join(lst)
        else:
            string = "0"
    else:
        lst = [
            f"({features[c]})**({round_d(w[c])})"
            for c in range(len(w))
            if round_d(w[c]) != 0
        ]
        if len(lst) != 0:
            string = " * ".join(lst)
        else:
            string = "1"

    if logspace:
        string += "*{}".format(str(round_d(eval(f"exp_int_no_nan({b})"))))
    else:
        string += "+{}".format(str(round_d(b)))
    return string


def _prepare_data(data, features, logspace):
    X, y = load_expected_post(data, features)
    if logspace:
        X = log_no_nan(X)
        y = log_no_nan(y)
    X = scaler.fit(X).transform(X)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y).view(-1, 1)
    return X, y


def model(x, w, b):
    return x @ w.t() + b


def criterion(y_pred, y, w, b):
    m = torch.nn.Softmin(dim=1)
    diff = torch.cat(
        (
            torch.square(y_pred[:, 0].view(-1, 1) - y),
            torch.square(y_pred[:, 1].view(-1, 1) - y),
            # torch.square(y_pred[:, 2].view(-1, 1) - y),
        ),
        dim=1,
    )
    soft = (y_pred * m(diff)).sum(dim=1).reshape(-1, 1)
    diff2 = soft - y
    return torch.sum(diff2 * diff2) / diff2.numel()
    # + 0.05 * (
    #     torch.sum(torch.abs(w)) + torch.sum(torch.abs(b))
    # )


def recover_wb(X, y_pred, features):
    min_val = scaler.data_min_[0]
    max_val = scaler.data_max_[0]
    X = X * (max_val - min_val) + min_val
    y = y_pred * (max_val - min_val) + min_val
    data1 = torch.cat((X, y[:, 0].view(-1, 1)), dim=1)
    points1 = data1[:8, :].tolist()
    data2 = torch.cat((X, y[:, 1].view(-1, 1)), dim=1)
    points2 = data2[:8, :].tolist()
    coefficients1, intercept1 = fit_line_or_hyperplane(points1)
    coefficients2, intercept2 = fit_line_or_hyperplane(points2)
    print(to_string(coefficients1, intercept1, features, False))
    print(to_string(coefficients2, intercept2, features, False))


def learn_reg_models(data, features, logspace):
    i = 0
    X, y = _prepare_data(data, features, logspace)
    w = torch.randn(2, len(features), dtype=torch.float32, requires_grad=True)
    b = torch.randn(2, dtype=torch.float32, requires_grad=True)
    y_pred = torch.randn(len(y), 1, dtype=torch.float32)
    # prev_loss = 100000
    while i < 800000:
        y_pred = model(X, w, b)
        loss = criterion(y_pred, y, w, b)
        loss.backward()
        with torch.no_grad():
            w -= w.grad * 0.01
            b -= b.grad * 0.01
            w.grad.zero_()
            b.grad.zero_()
        if i % 1000 == 0:
            print(f"{i}===>{loss.item()}")
            # if prev_loss - loss.item() < 0.005 and loss.item() < 1.5:
            # break
        i = i + 1

    recover_wb(X, y_pred, features)
    # min_val = scaler.data_min_[0]
    # max_val = scaler.data_max_[0]
    # # Recover original weights
    # print("w and b before recovery")
    # print(f"{w=}")
    # print(f"{b=}")
    # w = w / (max_val - min_val)
    # # Recover original bias
    # b = b - torch.sum(w * min_val)
    # w = w.detach().cpu().numpy()
    # b = b.detach().cpu().numpy()
    # print("w and b after recovery")
    # print(f"{w=}")
    # print(f"{b=}")
    # print(to_string(w[0], b[0], features, logspace))
    # print(to_string(w[1], b[1], features, logspace))
    # # print(to_string(w[2], b[2][0], features, logspace))
