import numpy as np

from metrikai import scan
from metrikai import SmartScan


def test_scan():
    assert list(scan(4, 1, 1)) == [0, 1, 2, 3]
    assert list(scan(4, 2, 1)) == [0, 1, 2]
    assert list(scan(4, 3, 1)) == [0, 1]
    assert list(scan(7, 3, 3)) == [0, 3, 4]
    assert list(scan(9, 5, 1)) == [0, 1, 2, 3, 4]
    assert list(scan(9, 5, 2)) == [0, 2, 4]
    assert list(scan(8, 4, 2)) == [0, 2, 4]


def test_smart_scan():
    im = np.zeros((10, 10))
    scanner = SmartScan(im, 5, 5)
    assert list(scanner.scanxy()) == []

    im[6:9, 6:9] = 255
    assert list(scanner.scanxy()) == [(5, 5)]
