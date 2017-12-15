import cv2 as cv
import numpy as np
import scipy.signal
import skimage.morphology
import skimage.filters


# colors
blue = (255, 0, 0)


def cleanup(im, remove_lines=False):
    im = cv.adaptiveThreshold(
        im,
        255,                             # max value
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        15,                              # gausian window size
        16,                              # adaptive threshold value
    )

    if remove_lines:
        # Remove horizontal and vertical lines
        kernel = skimage.morphology.rectangle(100, 1)
        mask = skimage.filters.median(im, kernel) | skimage.filters.median(im, kernel.T)
        im = im & ~mask

    return im


def window(iw, bw, x):
    pad = bw // 2
    if x <= pad:
        return 0, x
    elif iw - x <= pad:
        return iw - bw, bw - (iw - x)
    else:
        return x - pad, pad


def show(im, x, y, w, h):
    ih, iw = im.shape
    bw, bh = 600, 400
    bx, rx = window(iw, bw, x)
    by, ry = window(ih, bh, y)

    win = im[by:by + bh, bx:bx + bw]
    win = cv.cvtColor(win.copy(), cv.COLOR_GRAY2BGR)

    cv.rectangle(win, (rx, ry), (rx + w, ry + h), blue, 1)
    cv.imshow('im', win)


def show_samples(name, samples, w=40, s=25, grid=128):
    canvas = np.zeros(((samples.shape[0] // w + 1) * s, w * s))
    for j in range(samples.shape[0] // w + 1):
        for i in range(w):
            if j * w + i < samples.shape[0]:
                canvas[j * s:j * s + s, i * s:i * s + s] = samples[j * w + i].reshape((s, s))
                if grid is not None:
                    canvas[j * s:j * s + s, i * s:i * s + 1] = grid
                    canvas[j * s + s - 1:j * s + s, i * s:i * s + s] = grid
    cv.imshow(name, canvas)


def scan(iw, w, step):
    for x in range(0, iw - w + 1, step):
        yield x
    if (iw - w) > x:
        yield iw - w


def scanxy(im, w, step):
    ih, iw = im.shape
    for y in scan(ih, w, step):
        for x in scan(iw, w, step):
            yield x, y


class SmartScan:

    def __init__(self, im, w, step):
        self.im = im
        self.w = w
        self.step = step
        self.axes = [0, 0]

    def check(self):
        y, x = self.axes
        return self.im[y:y + self.w, x:x + self.w].mean() > 10

    def scan(self, axis):
        iw = self.im.shape[axis]

        while self.axes[axis] < iw - self.w + 1:
            yield self.axes[axis]

        if (iw - self.w) > self.axes[axis]:
            yield iw - self.w

    def scanxy(self):
        self.axes = [0, 0]
        for y in self.scan(0):
            self.axes[1] = 0
            for x in self.scan(1):
                if self.check():
                    yield x, y
                    self.axes[1] += self.step
                else:
                    self.axes[1] += self.w
            self.axes[0] += self.step


def line_scanxy(im, w, step):
    ih, iw = im.shape
    win = scipy.signal.hann(20)
    for x in range(0, iw - step, step):
        means = im[:, x:x + (w if x + w < iw else iw)].mean(axis=1)
        means = scipy.signal.convolve(means, win, mode='same') / sum(win)
        minimums = scipy.signal.argrelextrema(means, np.less_equal, order=10)[0]
        minimums = minimums[scipy.signal.convolve(np.diff(minimums), [.5, .5]) > 5]
        for y, h in zip(minimums, np.diff(minimums)):
            if x < iw - h + 1 and im[y:y + h, x:x + h].mean() > 10:
                yield x, y


def im_to_X(im):
    im = cv.resize(im, (25, 25), interpolation=cv.INTER_CUBIC)
    return im.reshape((1, 25 * 25))


def wait_for_key(key):
    while cv.waitKey(0) != key:
        pass


def rotate(i, n, d):
    j = i + d
    if j > n - 1 or j < 0:
        return j % n
    else:
        return j


def italic(slope, yintercept, w, h):
    if yintercept.size == 0:
        return np.array([], int), np.array([], int)
    y = np.arange(h)
    x = slope * y
    x = np.pad(x.reshape(1, h), [(0, yintercept.size - 1), (0, 0)], 'edge').T + yintercept
    x[x < 0] = w + x[x < 0]
    x[x > w - 1] = x[x > w - 1] % w
    return y, np.ceil(x.T).astype(int)


def move(a, *offset):
    return [x + o for x, o in zip(a, offset)]


def find_slope(im, axis=0):
    h, w = im.shape
    window = scipy.signal.blackman(5)
    slopes = []
    maxima_means = []
    for i in range(21):
        slope = i * -0.05
        medians = np.median(im[italic(slope, np.arange(w), w, h)], axis=axis)
        convolved = scipy.signal.convolve(medians, window, mode='same') / sum(window)
        maximas = scipy.signal.argrelextrema(convolved, np.greater, order=6)[0]
        slopes.append(slope)
        maxima_means.append(convolved[maximas].mean() if maximas.size else 0)
    return slopes[maxima_means.index(max(maxima_means))]


def find_lines(im, win, slope, axis=0):
    h, w = im.shape
    means = im[italic(slope, np.arange(w), w, h)].mean(axis=axis)
    means = scipy.signal.convolve(means, win, mode='same') / sum(win)
    minimas = scipy.signal.argrelextrema(means, np.less_equal, order=10)[0]
    return minimas[scipy.signal.convolve(np.diff(minimas), [.5, .5]) > 5]
