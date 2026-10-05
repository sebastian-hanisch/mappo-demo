"""Orakel: die angezeigte „erklärte Varianz“ des Kritikers im Training (`_explained_variance`) ist die übliche RL-Kennzahl
1 - Var(Ziel - Vorhersage) / Var(Ziel) (zentrierte Residuen, ein konstanter Versatz wird verziehen) und kein R².
sklearn.explained_variance_score ist dieselbe Definition; r2_score (SSE/SST) weicht bei Versatz bewusst ab."""
import numpy as np
import pytest

import mappo_ppo as ppo

metrics = pytest.importorskip("sklearn.metrics")


def test_explained_variance_matches_sklearn_definition_and_differs_from_r2_only_by_offset():
    rng = np.random.default_rng(0)
    for _ in range(50):
        target = rng.normal(size=(6, 8, 3)) * rng.uniform(0.5, 4.0)
        predicted = target + rng.normal(size=target.shape) * rng.uniform(0.1, 2.0) + rng.choice([0.0, 1.5])
        ours = ppo._explained_variance(target, predicted)
        flat_t, flat_p = target.ravel(), predicted.ravel()
        assert abs(ours - metrics.explained_variance_score(flat_t, flat_p)) < 1e-6
    constant_offset = target + 2.0
    assert abs(ppo._explained_variance(target, constant_offset) - 1.0) < 1e-6      # Versatz verziehen (Definition)
    assert metrics.r2_score(target.ravel(), constant_offset.ravel()) < 1.0 - 1e-3  # R² bestraft ihn
