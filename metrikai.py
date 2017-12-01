import cv2 as cv
import numpy as np


# colors
blue = (255, 0, 0)


def cleanup(im):
    return cv.adaptiveThreshold(
        im,
        255,                             # max value
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        15,                              # gausian window size
        16,                              # adaptive threshold value
    )


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


def im_to_X(im):
    im = cv.resize(im, (25, 25), interpolation=cv.INTER_CUBIC)
    return im.reshape((1, 25 * 25))


def wait_for_key(key):
    while cv.waitKey(0) != key:
        pass
