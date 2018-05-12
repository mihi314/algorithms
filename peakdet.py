import numpy as np
import matplotlib.pyplot as plt
from pytest import approx


def peakdet(y, delta):
    """
    Return the indices (indices_min, indices_max) of local maxima and minima
    ("peaks") in the vector y.
    
    A point is considered a maximum if it locally has the maximal value, and is
    preceded and followed by a value lower by delta.
    Based on http://billauer.co.il/peakdet.html.
    """
    y = np.asarray(y)

    if not len(y.shape) == 1:
        raise ValueError("y has to be one-dimensional")
    delta = float(delta)
    if delta <= 0:
        raise ValueError("delta must be a positive number")
    
    # try starting with search for max peaks or min peaks first
    # take whichever returns more peaks
    # todo: could be optimized to not completely go through the array twice
    mins1, maxes1, _ = _peakdet(y, delta, True)
    mins2, maxes2, _ = _peakdet(y, delta, False)
    if len(mins1) + len(maxes1) > len(mins2) + len(maxes2):
        return mins1, maxes1
    else:
        return mins2, maxes2


def peakdet_wrapped(y, delta):
    """
    Return the indices (indices_min, indices_max) of local maxima and minima
    ("peaks") in the vector y. y is interpreted to wrap around the ends, e.g.
    to be a distribution over angles.
    
    A point is considered a maximum if it locally has the maximal value, and is
    preceded and followed by a value lower by delta.
    Based on http://billauer.co.il/peakdet.html.
    """
    y = np.asarray(y)

    if not len(y.shape) == 1:
        raise ValueError("y has to be one-dimensional")
    delta = float(delta)
    if delta <= 0:
        raise ValueError("delta must be a positive number")
    
    mins1, maxes1, state = _peakdet(y, delta, True)
    # start again from the beginning with the previous state so that peaks wrapped around
    # the ends are now also detected
    mins2, maxes2, state = _peakdet(y, delta, True, state)
    return mins2, maxes2


def _peakdet(y, delta, lookformax, state=None):
    """
    Returns (indices_min, indices_max, state). Starts with looking for max when
    lookformax == True, else min. Suppresses first extremum.
    """
    indices_min = []
    indices_max = []
    
    if state:
        mn, mx, mnpos, mxpos, lookformax, first_peak = state
    else:
        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN
        first_peak = True

    for i, this in enumerate(y):
        # keep track of the current max and min
        if this > mx:
            mx = this
            mxpos = i
        if this < mn:
            mn = this
            mnpos = i

        if lookformax and this < mx-delta:
            # we have fallen delta below the current max, this is now a legitimate max,
            # start looking for a min now
            if not first_peak:
                indices_max.append(mxpos)
            mnpos, mn = i, this
            lookformax = False
            first_peak = False
        elif not lookformax and this > mn+delta:
            # we have risen delta above the current min, this is now a legitimate min,
            # start looking for a max now
            if not first_peak:
                indices_min.append(mnpos)
            mxpos, mx = i, this
            lookformax = True
            first_peak = False
    
    state = (mn, mx, mnpos, mxpos, lookformax, first_peak)
    return np.array(indices_min), np.array(indices_max), state


def get_testdata(mus):
    xs = np.linspace(0, 360, 1000)
    sigma = 50
    ys = np.zeros_like(xs)
    for mu in mus:
        for k in [-1, 0, 1]:
            ys += 100*np.exp(-(xs-mu + 360*k)**2/sigma**2)
    return xs, ys

def check_peakdet(fun, mus, mins_should, maxes_should):
    xs, ys = get_testdata(mus)
    mins, maxes = fun(ys, delta=50)
    np.testing.assert_allclose(sorted(xs[mins]) if len(mins) else [], sorted(mins_should), atol=1)
    np.testing.assert_allclose(sorted(xs[maxes]) if len(maxes) else [], sorted(maxes_should), atol=1)

def test_normal():
    check_peakdet(peakdet, [20], [200], [])
    check_peakdet(peakdet, [100], [], [100])
    check_peakdet(peakdet, [50, 250], [150], [50, 250])
    check_peakdet(peakdet, [], [], [])
    check_peakdet(peakdet, [100, 180], [], [110])

def test_wrapped():
    check_peakdet(peakdet_wrapped, [20], [200], [20])
    check_peakdet(peakdet_wrapped, [100], [280], [100])
    check_peakdet(peakdet_wrapped, [50, 250], [150, 330], [50, 250])
    check_peakdet(peakdet_wrapped, [], [], [])
    check_peakdet(peakdet_wrapped, [100, 180], [320], [110])


if __name__ == "__main__":
    xs, ys = get_testdata([100, 180])
    # ys += np.random.normal(size=ys.shape)
    mins, maxes = peakdet_wrapped(ys, delta=50)

    plt.plot(xs, ys)
    if len(maxes):
        plt.plot(xs[maxes], ys[maxes], "rx")
    if len(mins):
        plt.plot(xs[mins], ys[mins], "bx")
    plt.show()
