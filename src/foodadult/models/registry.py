"""Model registry and factory for all supervised baseline and tuned classifiers."""

from typing import Any, Dict, List
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from foodadult.features.preprocess import build_preprocessor

MODEL_REGISTRY = {
    "dummy": DummyClassifier,
    "rf": RandomForestClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "dt": DecisionTreeClassifier,
    "gb": GradientBoostingClassifier,
}


def get_available_models() -> List[str]:
    """Return list of supported model keys."""
    return list(MODEL_REGISTRY.keys())


def get_model(model_key: str, config: Dict[str, Any], **override_params) -> Pipeline:
    """Build and return complete Pipeline (preprocessor + classifier) for model_key.

    Args:
        model_key: Key in MODEL_REGISTRY ('dummy', 'rf', 'svm', 'knn', 'dt', 'gb').
        config: Configuration dictionary.
        override_params: Additional keyword arguments passed to classifier constructor.

    Returns:
        Scikit-Learn Pipeline ready for fit / cross_validate.

    Raises:
        KeyError: If model_key is not in MODEL_REGISTRY.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{model_key}'. Available: {get_available_models()}"
        )

    random_state = int(config.get("random_state", 42))
    class_weight = config.get("class_weight", None)

    # Base parameters per model
    if model_key == "dummy":
        params = {"strategy": "most_frequent"}
    elif model_key == "rf":
        params = {"random_state": random_state, "n_jobs": -1}
        if class_weight:
            params["class_weight"] = class_weight
    elif model_key == "svm":
        params = {"kernel": "rbf", "probability": True, "random_state": random_state}
        if class_weight:
            params["class_weight"] = class_weight
    elif model_key == "knn":
        params = {}
    elif model_key == "dt":
        params = {"criterion": "gini", "random_state": random_state}
        if class_weight:
            params["class_weight"] = class_weight
    elif model_key == "gb":
        params = {"random_state": random_state}
    else:
        params = {}

    params.update(override_params)
    estimator_cls = MODEL_REGISTRY[model_key]
    classifier = estimator_cls(**params)

    # Combine with leak-free preprocessor
    preprocessor = build_preprocessor(config)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", classifier),
    ])
    return pipeline
