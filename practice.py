import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
from sympy import symbols, gcd
import numpy as np
from numpy.linalg import svd


def fit_plane(points):
    # Ensure we have three points
    if len(points) != 3 or len(points[0]) != 3:
        raise ValueError(
            "fit_plane requires exactly three points in three-dimensional space."
        )

    # Extract coordinates of the points
    x1, y1, z1 = points[0]
    x2, y2, z2 = points[1]
    x3, y3, z3 = points[2]

    # Calculate vectors in the plane
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x3 - x1, y3 - y1, z3 - z1])

    # Calculate the normal vector to the plane
    normal_vector = np.cross(v1, v2)
    print(f"{normal_vector=}")

    # Calculate the coefficients of the plane equation
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    a, b, c = normal_vector
    d = -(a * x1 + b * y1 + c * z1)

    return a, b, c, d


# Example with three points (plane)
points_plane = np.array([[1, -2, 1], [4, -2, -2], [4, 1, 4]])
coefficients_plane = fit_plane(points_plane)
print("Plane Coefficients:", coefficients_plane)


######################################################################################################################
# def loss(y, y_pred, threshold):
#     res = [(y1 - y2) ** 2 for y1, y2 in zip(y, y_pred)]
#     indices = [i for i, element in enumerate(res) if element <= threshold]
#     return indices


# def remove_samples(X, y, y_pred, threshold):
#     idx = loss(y, y_pred, threshold)
#     print(f"len_idx{len(idx)}")
#     X = np.delete(X, idx, axis=0)
#     y = np.delete(y, idx)
#     print(f"len_X{len(X)}")
#     print(f"len_y{len(y)}")
#     return X, y


# def learn_reg_models(X_init, Y_post):
#     models = []
#     threshold = 6
#     # reg = LinearRegression(fit_intercept=True)
#     # reg.fit(X_init, Y_post)
#     # y_pred = reg.predict(X_init)
#     # print(f"{features=}")
#     # print(reg.coef_)
#     # print(reg.intercept_)
#     # print(to_string(reg, features, logspace))
#     while len(X_init) > 0:
#         print(len(X_init))
#         X, y = X_init, Y_post
#         reg = LinearRegression(fit_intercept=True)
#         # reg = linear_model.Lasso(alpha=0.001)
#         reg.fit(X, y)
#         y_pred = reg.predict(X)
#         plt.plot(X, y_pred, color="r")
#         X_init, Y_post = remove_samples(X, y, y_pred, threshold)
#         if len(X_init) < len(X):
#             print(f"This is coeff::{reg.coef_}")
#             print(f"This is intercept::{reg.intercept_}")
#             # print("model inserted")
#             models.append(reg)
#         else:
#             threshold += 2
#     print(f"set of all models: {len(models)}")
#     # for model in models:
#     #     print(model.coef_)
#     #     print(model.intercept_)
#     #     print(f"Next Model: \n")


# X1 = np.arange(-2, 2, 0.1).reshape(-1, 1)
# Y1 = -8 * X1 + 2 + 0.4 * np.random.randn(*X1.shape)
# X2 = np.arange(-10, -6, 0.1).reshape(-1, 1)
# Y2 = 5 * X2 + 3 + 0.4 * np.random.randn(*X2.shape)
# res = np.hstack((np.vstack((X1, X2)), np.vstack((Y1, Y2))))
# np.random.shuffle(res)
# X_init, Y_post = res[:, 0].reshape(-1, 1), res[:, 1].reshape(-1, 1)

# # Plot Y1
# plt.scatter(X_init, Y_post, label="Y1 & Y2", color="b")
# learn_reg_models(X_init, Y_post)
# # # Plot Y2
# # plt.scatter(X2, Y2, label="Y2", color="r")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.title("Scatter Plot of Y1 and Y2")
# plt.show()


################################## SoftMin #################################################
# scaler = MinMaxScaler()
# X1 = np.arange(-2, 2, 0.1).reshape(-1, 1)
# Y1 = -8 * X1 + 2 + 0.4 * np.random.randn(*X1.shape)
# # X1, Y1 = scaler.fit(X1).transform(X1), scaler.fit(Y1).transform(Y1)
# X2 = np.arange(-10, -6, 0.1).reshape(-1, 1)
# Y2 = 5 * X2 + 3 + 0.4 * np.random.randn(*X2.shape)
# # X2, Y2 = scaler.fit(X2).transform(X2), scaler.fit(Y2).transform(Y2)
# res = np.hstack((np.vstack((X1, X2)), np.vstack((Y1, Y2))))
# np.random.shuffle(res)
# X, y = res[:, 0].reshape(-1, 1), res[:, 1].reshape(-1, 1)
# X = scaler.fit(X).transform(X)
# X = torch.from_numpy(X)
# y = torch.from_numpy(y)
# w = torch.randn(2, 1, dtype=torch.double, requires_grad=True)
# b = torch.randn(2, dtype=torch.double, requires_grad=True)


# def criterion(y_pred, y):
#     m = torch.nn.Softmin(dim=1)
#     diff = torch.cat(
#         (
#             torch.square(y_pred[:, 0].view(-1, 1) - y),
#             torch.square(y_pred[:, 1].view(-1, 1) - y),
#         ),
#         dim=1,
#     )
#     soft = (y_pred * m(diff)).sum(dim=1).reshape(-1, 1)
#     diff2 = soft - y
#     return torch.sum(diff2 * diff2) / diff2.numel()


# def criterion(y_pred, y):
#     m = torch.nn.Softmin(dim=1)
#     # output = m(y_pred)
#     # output, _ = torch.min(m(y_pred), dim=1)
#     # _, indices = torch.max(m(y_pred), dim=1)
#     # output = y_pred.gather(1, indices.view(-1, 1))
#     output = (y_pred * m(y_pred)).sum(dim=1).reshape(-1, 1)
#     diff = output - y
#     # param = torch.cat((w, b.view(-1, 1)), dim=1)
#     # + torch.nn.functional.cosine_similarity(
#     #     param[0, :], param[1, :], dim=0
#     # )
#     return torch.sum(diff * diff) / diff.numel()


# def model(x):
#     return x @ w.t() + b


# for i in range(100000):
#     y_pred = model(X)
#     loss = criterion(y_pred, y)
#     loss.backward()
#     with torch.no_grad():
#         w -= w.grad * 0.01
#         b -= b.grad * 0.01
#         w.grad.zero_()
#         b.grad.zero_()
#     if i % 1000 == 0:
#         print(f"{i}===>{loss.item()}")

# print(w)
# print(b)

# min_val = scaler.data_min_[0]
# max_val = scaler.data_max_[0]
# # Recover original weights
# original_weights = w / (max_val - min_val)
# # Recover original bias
# original_bias = b.view(-1, 1) - original_weights * min_val
# print(f"{original_weights=}")
# print(f"{original_bias=}")

# tensor([[0.3326],
#         [0.2277]], dtype=torch.float64, requires_grad=True)
# tensor([-0.5144,  1.0579], dtype=torch.float64, requires_grad=True)
