from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from scipy.optimize import least_squares

from peakdet import peakdet


def loop_the_loop(A, B, C, loop_size=2, N=100):
    """
    Calculate a curve that goes from point A to C, then loops and finishes off with C to B.
    loop_size <= 1 is no loop, loop_size > 1 is loopy.
    Returns an array of points with shape (n, len(A)).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)

    # n control points
    ctrl = np.array([A, C + loop_size*(C-A), C + loop_size*(C-B), B])
    # n+k+1? knots
    # t = [0, 0, 0, 1, 1, 1]
    t = np.linspace(0, 1, ctrl.shape[0]-2)
    t = np.append([0, 0, 0], t)
    t = np.append(t, [1, 1, 1])

    spline = interp.BSpline(t, ctrl, k=3, axis=0)
    u = np.linspace(0, 1, N)
    out = interp.splev(u, spline)
    return out

def norm(vec):
    return np.sqrt(np.sum(vec*vec, axis=-1))

def cross(a, b):
    return a[...,0]*b[...,1] - a[...,1]*b[...,0]

def line_point_dist(lines, ps):
    """
    Closest distance of a point to an infinite line defined by two points (a, b).
    The arguments can also be lists of lines and points, in that case the distance for
    each combination is returned, with shape lines.shape[:-2] + ps.shape[:-1].
    """
    lines = np.asarray(lines)
    ps = np.asarray(ps)
    assert(lines.shape[-2:] == (2, 2))
    assert(ps.shape[-1] == 2)

    a = lines[...,0,:]
    b = lines[...,1,:]
    for _ in range(max(len(ps.shape)-1, 1)):
        a = np.expand_dims(a, -2)
        b = np.expand_dims(b, -2)

    v_hat = (b - a) / np.expand_dims(norm(b - a), -1)
    d_normal = np.abs(cross(v_hat, ps - a))
    return d_normal

def approximate_polyline(polyline, epsilon):
    """Return the indices of the points that define the vertices of the approximated polyline."""
    # see: https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
    a_idx = 0
    b_idx = len(polyline) - 1
    a = polyline[a_idx,:]
    b = polyline[b_idx,:]

    ds = line_point_dist([a, b], polyline)
    furthest = np.argmax(ds)
    if ds[furthest] >= epsilon:
        polyline_left = approximate_polyline(polyline[:furthest+1,:], epsilon)
        polyline_right = approximate_polyline(polyline[furthest:,:], epsilon)
        return np.concatenate([polyline_left[:-1], furthest + polyline_right])
    else:
        return np.array([a_idx, b_idx])

def line_line_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    # from: https://stackoverflow.com/a/42727584
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return np.inf, np.inf
    return x/z, y/z

def plot_line(theta, rho):
    alpha = np.cos(theta)
    beta = np.sin(theta)
    x0 = alpha * rho
    y0 = beta * rho

    l = 1
    x1 = x0 + l*(-beta)
    y1 = y0 + l*(alpha)
    x2 = x0 - l*(-beta)
    y2 = y0 - l*(alpha)

    plt.plot([x1, x2], [y1, y2])

def robust_line_fit(points):
    """Robustly fit a line to points, and return (theta, rho)."""
    weights = np.ones(points.shape[0])
    p0 = [1, 0]
    f_scale = np.NaN

    for loss in ["linear", "huber"]:
        def residual_fun(params):
            theta, rho = params
            r = np.cos(theta)*points[:,0] + np.sin(theta)*points[:,1] - rho
            return weights * r

        res = least_squares(residual_fun, p0, loss=loss, f_scale=f_scale)
        p0 = res.x

        residuals = res.fun
        robust_std = np.median(np.abs(residuals)) / 0.6745
        f_scale = robust_std
        # weights = np.where(np.abs(residuals) < robust_std, (1 - (residuals/robust_std)**2)**2, 0)
    return res.x

def project_point_onto_line(theta, rho, point):
    alpha = np.cos(theta)
    beta = np.sin(theta)
    a = rho * np.array([alpha, beta])
    v = np.array([-beta, alpha])
    return a + (point - a).dot(v) * v

def replace_slices(x, slices):
    """
    slices is a list of Slice(i, j, x).
    Replaces x[slice.i:slice.j] in the orignial x with slice.x. slice.x does not
    need to have the same length as the replaced part.
    """
    parts = []
    idx = 0
    for sl in sorted(slices, key=lambda sl: sl.i):
        assert(sl.i <= sl.j)
        parts.append(x[idx:sl.i])
        parts.append(sl.x)
        idx = sl.j
    parts.append(x[idx:])
    return np.concatenate(parts)

def calligraphic_fit_old(points, loopiness=2):
    polyline = approximate_polyline(points, 0.03)

    for a_idx, b_idx in zip(polyline, polyline[1:]):
        theta, rho = robust_line_fit(points[a_idx:b_idx+1,:])
        a = project_point_onto_line(theta, rho, points[a_idx,:])
        b = project_point_onto_line(theta, rho, points[b_idx,:])
        plt.plot(*np.array([a,b]).T, "r")
    
    return points[polyline,:]

def calligraphic_fit(points, loopiness):
    """Adds `loopiness` numer of loops along the curve where the curvature is highest."""
    CURVATURE_FLAT = 1
    LOOP_SIZE = 8

    ## normalize the points so that the paramters and constatns make more sense across different datasets
    points_means = points.mean(axis=0, keepdims=True)
    points_stds = points.std(axis=0, keepdims=True)
    points = (points - points_means) / points_stds


    ## fit a spline
    num_points = points.shape[0]
    u = np.linspace(0, 1, num_points)

    k = 3
    # the number of control points is about one third of the original number of points
    # determines the extend of the smoothing
    t = np.linspace(0, 1, num_points // 3)
    t = np.concatenate([[0]*k, t, [1]*k])

    spl = interp.make_lsq_spline(u, points, t, k=k)


    ## calc curvature and find extrema
    u2 = np.linspace(0, 1, num_points*10)
    x = interp.splev(u2, spl)
    xdot = interp.splev(u2, spl, 1)
    xddot = interp.splev(u2, spl, 2)
    # see: https://www.math.tugraz.at/~wagner/Dreibein
    curvature = cross(xdot, xddot) / norm(xdot)**3

    mins, maxes = peakdet(curvature, delta=0)


    ## add `loopiness` number of loops where the curvature is highest
    peaks = np.concatenate([mins, maxes])
    peaks = sorted(peaks, key=lambda p: abs(curvature[p]), reverse=True)

    slices = []
    Slice = namedtuple("Slice", ["i", "j", "x"])
    for peak in peaks[:loopiness]:
        # find the first points left and right of the peak that have a low enough curvature
        for i in reversed(range(0, peak)):
            if np.sign(curvature[peak]) * curvature[i] <= CURVATURE_FLAT:
                break
        for j in range(peak+1, len(curvature)):
            if np.sign(curvature[peak]) * curvature[j] <= CURVATURE_FLAT:
                break
        
        # extend the loop out from these low-curvature points
        C = line_line_intersect(x[i-1], x[i], x[j], x[j+1])
        # todo: maybe handle the case when there is no intersection
        loop = loop_the_loop(x[i], x[j], C, LOOP_SIZE)
        slices.append(Slice(i, j+1, loop))

    x_new = replace_slices(x, slices)


    ## undo the normalization
    x_new = x_new * points_stds + points_means
    return x_new


def main():
    data = np.loadtxt("calligraphic_fit/data.txt", skiprows=2)
    data = data[10:-10,1:]
    # x = 0.5584 * x + 12.664

    data_loopy = calligraphic_fit(data, loopiness=5)
    plt.plot(*data_loopy.T)
    plt.plot(*data.T)
    plt.show()

if __name__ == "__main__":
    main()
