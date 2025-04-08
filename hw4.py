import numpy as np
from scipy import integrate

# 被積分函數 f(x, y)
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# y的上下限
def y_lower(x):
    return np.sin(x)

def y_upper(x):
    return np.cos(x)

#  (a) Trapezoid Rule for n=4,m=4
n = m = 4
x_vals = np.linspace(0, np.pi/4, n+1)
trapz_result = 0

for x in x_vals:
    y_vals = np.linspace(y_lower(x), y_upper(x), m+1)
    f_vals = f(x, y_vals)
    trapz_result += np.trapezoid(f_vals, y_vals)

dx = (np.pi / 4) / n
trapz_result *= dx

#(b) Gaussian Quadrature (3x3)
# 3-point Gauss-Legendre nodes & weights
gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(3)

# 二維 Gaussian Quadrature
def gauss_quad_2d(f, ax, bx, ay_func, by_func):
    result = 0.0
    for i in range(3):
        xi = 0.5 * (bx - ax) * gauss_nodes[i] + 0.5 * (bx + ax)
        wx = 0.5 * (bx - ax) * gauss_weights[i]
        ay = ay_func(xi)
        by = by_func(xi)
        for j in range(3):
            yj = 0.5 * (by - ay) * gauss_nodes[j] + 0.5 * (by + ay)
            wy = 0.5 * (by - ay) * gauss_weights[j]
            result += wx * wy * f(xi, yj)
    return result

gauss_result = gauss_quad_2d(f, 0, np.pi/4, y_lower, y_upper)

#(c) 精確值（scipy dblquad）
exact_val, _ = integrate.dblquad(f, 0, np.pi/4, y_lower, y_upper)

#結果輸出
print("Trapezoid Rule (n=4, m=4):", trapz_result)
print("Gaussian Quadrature (n=3, m=3):", gauss_result)
print("Exact Value:", exact_val)
