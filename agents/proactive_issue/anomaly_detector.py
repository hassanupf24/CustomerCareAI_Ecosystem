"""
Anomaly Detector — Isolation Forest / statistical anomaly detection.

Detects anomalies in customer account usage data to predict potential
issues before they are reported.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


def detect_anomalies(
    usage_logs: list[dict],
    threshold: float = -0.5,
) -> list[dict]:
    """
    Detect anomalies in usage logs.

    Each usage log is expected to have numeric fields like:
    - api_calls, error_count, latency_ms, login_failures, etc.

    Returns:
        List of anomaly dicts with keys: index, anomaly_score, fields
    """
    if not usage_logs:
        return []

    # Extract numeric features
    numeric_keys = _extract_numeric_keys(usage_logs)
    if not numeric_keys:
        return []

    feature_matrix = _build_feature_matrix(usage_logs, numeric_keys)

    try:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
        )
        scores = model.fit(feature_matrix).decision_function(feature_matrix)
        predictions = model.predict(feature_matrix)

        anomalies = []
        for i, (score, pred) in enumerate(zip(scores, predictions)):
            if pred == -1:  # anomaly
                anomalies.append({
                    "index": i,
                    "anomaly_score": round(float(score), 4),
                    "data": usage_logs[i],
                })

        return anomalies
    except ImportError:
        logger.warning("sklearn_not_available_using_statistical_fallback")
        return _statistical_fallback(usage_logs, numeric_keys, feature_matrix)
    except Exception as e:
        logger.error("anomaly_detection_failed", error=str(e))
        return _statistical_fallback(usage_logs, numeric_keys, feature_matrix)


def _extract_numeric_keys(logs: list[dict]) -> list[str]:
    """Find all keys with numeric values across logs."""
    keys = set()
    for log in logs:
        for k, v in log.items():
            if isinstance(v, (int, float)):
                keys.add(k)
    return sorted(keys)


def _build_feature_matrix(logs: list[dict], keys: list[str]) -> np.ndarray:
    """Build a 2D numpy array from selected numeric keys."""
    rows = []
    for log in logs:
        row = [float(log.get(k, 0.0)) for k in keys]
        rows.append(row)
    return np.array(rows, dtype=np.float64)


def _statistical_fallback(
    logs: list[dict],
    keys: list[str],
    matrix: np.ndarray,
) -> list[dict]:
    """Z-score based anomaly detection as fallback."""
    if matrix.shape[0] < 3:
        return []

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-9
    z_scores = np.abs((matrix - mean) / std)
    max_z = z_scores.max(axis=1)

    anomalies = []
    for i, z in enumerate(max_z):
        if z > 2.0:
            anomalies.append({
                "index": i,
                "anomaly_score": round(float(-z), 4),
                "data": logs[i],
            })
    return anomalies
