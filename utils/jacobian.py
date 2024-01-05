import numpy as np

def dx_y(f, h, mode = "x"):
    if mode.lower() == "y":
        f = f.T
    elif mode.lower() == "x":
        f = f
    else:
        raise ValueError(f"false mode of dx_y: {mode}")
    
    a1 = 1.0 / (60.0 * h)
    a2 = -3.0 / (20.0 * h)
    a3 = 3.0 / (4.0 * h)
    b1 = 8.0 / (12.0 * h)
    b2 = 1.0 / (12.0 * h)

    ff = np.zeros_like(f)
    ff[:, 3:-3] = a1 * (f[:, 6:] - f[:, :-6]) + \
                 a2 * (f[:, 5:-1] - f[:, 1:-5]) + \
                 a3 * (f[:, 4:-2] - f[:, 2:-4])

    ff[:, 0] = (-3.0 * f[:, 0] + 4.0 * f[:, 1] - f[:, 2]) / (2.0 * h)
    ff[:, 1] = (-2.0 * f[:, 0] - 3.0 * f[:, 1] + 6.0 * f[:, 2] - f[:, 3]) / (6.0 * h)
    ff[:, 2] = b1 * (f[:, 3] - f[:, 1]) - b2 * (f[:, 4] - f[:, 0])

    ff[:, -3] = b1 * (f[:, -2] - f[:, -4]) - b2 * (f[:, -1] - f[:, -5])
    ff[:, -2] = (f[:, -4] - 6.0 * f[:, -3] + 3.0 * f[:, -2] + 2.0 * f[:, -1]) / (6.0 * h)
    ff[:, -1] = (f[:, -3] - 4.0 * f[:, -2] + 3.0 * f[:, -1]) / (2.0 * h)

    return ff.T if mode.lower() == "y" else ff

def jac(xx, yy):
    ny, nx = xx.shape
    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)
    
    xk = dx_y(xx, hx, "x")
    yk = dx_y(yy, hx, "x")
    xi = dx_y(xx, hy, "y")
    yi = dx_y(yy, hy, "y")

    Ajac = xk * yi - xi * yk
    Ajac = 1.0 / Ajac
    Akx = Ajac * yi
    Aky = -Ajac * xi
    Aix = -Ajac * yk
    Aiy = Ajac * xk

    jac_k_i = np.array([[Akx, Aky], [Aix, Aiy]])
    return jac_k_i


