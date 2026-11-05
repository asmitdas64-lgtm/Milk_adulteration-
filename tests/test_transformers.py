import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from foodadult.features.transformers import IQRClipper


def test_iqr_clipper_learns_bounds_and_clips():
    # 5 values: 10, 20, 30, 40, 50
    # Q1 = 20, Q3 = 40, IQR = 20
    # factor = 1.5 -> lower = 20 - 30 = -10, upper = 40 + 30 = 70
    X_train = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
    clipper = IQRClipper(factor=1.5)
    clipper.fit(X_train)

    assert clipper.lower_[0] == pytest.approx(-10.0)
    assert clipper.upper_[0] == pytest.approx(70.0)

    # Test values: inside (-5.0, 35.0), below (-50.0), above (100.0)
    X_test = np.array([[-5.0], [35.0], [-50.0], [100.0]])
    transformed = clipper.transform(X_test)

    # Values within bounds stay unchanged
    assert transformed[0, 0] == pytest.approx(-5.0)
    assert transformed[1, 0] == pytest.approx(35.0)
    # Extreme values are clipped to bounds
    assert transformed[2, 0] == pytest.approx(-10.0)
    assert transformed[3, 0] == pytest.approx(70.0)


def test_iqr_clipper_raises_when_not_fitted():
    clipper = IQRClipper(factor=1.5)
    with pytest.raises(NotFittedError):
        clipper.transform(np.array([[10.0]]))

