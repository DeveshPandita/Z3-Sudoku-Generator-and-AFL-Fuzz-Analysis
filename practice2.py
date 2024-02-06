import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# def fit_plane(points):
#     # Ensure we have three points
#     if len(points) != 3 or len(points[0]) != 3:
#         raise ValueError(
#             "fit_plane requires exactly three points in three-dimensional space."
#         )

#     # Extract coordinates of the points
#     x1, y1, z1 = points[0]
#     x2, y2, z2 = points[1]
#     x3, y3, z3 = points[2]

#     # Calculate vectors in the plane
#     v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
#     v2 = np.array([x3 - x1, y3 - y1, z3 - z1])

#     # Calculate the normal vector to the plane
#     normal_vector = np.cross(v1, v2)

#     # Calculate the coefficients of the plane equation
#     a, b, c = normal_vector
#     d = -(a * x1 + b * y1 + c * z1)

#     return a, b, c, d


# Example with three points (plane)
# points_plane = np.array([[1, -2, 1], [4, -2, -2], [4, 1, 4]])
# coefficients_plane = fit_plane(points_plane)
# print("Plane Coefficients:", coefficients_plane)


def fit_line_or_hyperplane(points):
    if len(points) == 2 and len(points[0]) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept
    elif len(points) >= 3 and len(points[0]) >= 2:
        centroid = np.mean(points, axis=0)
        u, s, vh = np.linalg.svd(points - centroid, full_matrices=False)
        normal = vh[-1]
        # print(f"{normal=}")
        coefficients = normal / np.linalg.norm(normal)
        intercept = -np.dot(coefficients, centroid)
        return coefficients, intercept
    else:
        raise ValueError("Invalid number of points or dimensions")


# points_hyperplane = np.array([[1, -2, 1], [4, -2, -2], [4, 1, 4]])
# coefficients, intercept = fit_line_or_hyperplane(points_hyperplane)
# print("Line or Hyperplane Coefficients:", coefficients)
# print("Line or Hyperplane Intercept (b):", intercept)

# # Example with two points (line)
# points_line = np.array([[1, 3], [2, 5]])
# slope, intercept = fit_line_or_hyperplane(points_line)
# print("\nLine or Hyperplane Slope:", slope)
# print("Line or Hyperplane Intercept:", intercept)


####################################################Do Not Remove This############################################
# X1 = np.arange(0, 4, 0.1).reshape(-1, 1)
# Y1 = 1 * X1 - 5 + 0.4 * np.random.randn(*X1.shape)

# X2 = np.arange(4, 8, 0.1).reshape(-1, 1)
# Y2 = -1 * X2 + 3 + 0.4 * np.random.randn(*X2.shape)
# res = np.hstack((np.vstack((X1, X2)), np.vstack((Y1, Y2))))
# np.random.shuffle(res)
# X, y = res[:, 0].reshape(-1, 1), res[:, 1].reshape(-1, 1)
# Y1_pred = 0.9673 * X1 - 4.9994
# Y2_pred = -0.9535 * X2 + 2.5576
# plt.scatter(X1, Y1, label="Line One", color="b")
# plt.scatter(X2, Y2, label="Line Two", color="y")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.plot(X1, Y1_pred, color="r")
# plt.plot(X2, Y2_pred, color="g")
# plt.legend()
# # plt.show()
# plt.savefig("second.png")
# plt.clf()
# # w = tensor([[0.9673], [-0.9535]], dtype=torch.float64, requires_grad=True)
# # b = tensor([-4.9994, 2.5576], dtype=torch.float64, requires_grad=True)

####################################################################################

# X1 = np.arange(-2, 2, 0.1).reshape(-1, 1)
# Y1 = -8 * X1 + 2 + 0.4 * np.random.randn(*X1.shape)
# X2 = np.arange(-10, -6, 0.1).reshape(-1, 1)
# Y2 = 5 * X2 + 3 + 0.4 * np.random.randn(*X2.shape)
# res = np.hstack((np.vstack((X1, X2)), np.vstack((Y1, Y2))))
# np.random.shuffle(res)
# X, y = res[:, 0].reshape(-1, 1), res[:, 1].reshape(-1, 1)
# Y1_pred = -7.9673 * X1 + 2.0029
# Y2_pred = 5.0454 * X2 + 3.2782
# plt.scatter(X1, Y1, label="Line One", color="b")
# plt.scatter(X2, Y2, label="Line Two", color="y")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.plot(X1, Y1_pred, color="r")
# plt.plot(X2, Y2_pred, color="g")
# plt.legend()
# # plt.show()
# plt.savefig("fist.png")
# # w = tensor([[5.0454], [-7.9673]], dtype=torch.float64, requires_grad=True)
# # b = tensor([3.2782, 2.0029], dtype=torch.float64, requires_grad=True)
