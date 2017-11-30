import cv2 as cv


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


def scan(iw, w, step):
    for x in range(0, iw - w + 1, step):
        yield x
    if (iw - w) > x:
        yield iw - w


def scanxy(im, w, step):
    ih, iw = im.shape
    for y in scan(ih, w, step):
        for x in scan(iw, w, 25):
            yield x, y


def im_to_X(im):
    im = cv.resize(im, (25, 25), interpolation=cv.INTER_CUBIC)
    return im.reshape((1, 25 * 25))


def wait_for_key(key):
    while cv.waitKey(0) != key:
        pass
