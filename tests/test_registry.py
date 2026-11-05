import pytest
from sklearn.pipeline import Pipeline
from foodadult.models.registry import get_model, get_available_models


@pytest.fixture
def mock_cfg():
    return {
        "random_state": 42,
        "class_weight": None,
        "data": {
            "features": ["ph", "temprature", "colour"],
        },
        "preprocess": {
            "imputer": "median",
            "outliers": "clip",
            "iqr_factor": 1.5,
            "scaler": "standard",
        }
    }


def test_registry_contains_all_six_models():
    expected = {"dummy", "rf", "svm", "knn", "dt", "gb"}
    assert set(get_available_models()) == expected


@pytest.mark.parametrize("model_key", ["dummy", "rf", "svm", "knn", "dt", "gb"])
def test_get_model_returns_full_pipeline(mock_cfg, model_key):
    pipeline = get_model(model_key, mock_cfg)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "clf" in pipeline.named_steps


def test_get_model_unknown_key_raises(mock_cfg):
    with pytest.raises(KeyError, match="Unknown model key"):
        get_model("xgboost", mock_cfg)

