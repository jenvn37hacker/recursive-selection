from dataclasses import replace
import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from recursive_selection.evaluation import PairedClusterBootstrap


@pytest.fixture
def paired_data():
    clusters = np.repeat(["a", "b", "c", "d"], [3, 1, 2, 4])
    y = np.array([0] * 4 + [1] * 6)
    reference = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    candidate = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    return y, reference, candidate, clusters


def test_batched_cluster_f1_matches_expanded_rows_and_shared_weights(paired_data):
    y, ref, cand, clusters = paired_data
    comparison = PairedClusterBootstrap(n_resamples=17, batch_size=5, random_state=42).compare(
        y, ref, cand, clusters, strata=y,
    )
    rng = np.random.default_rng(42)
    cluster_names, codes = np.unique(clusters, return_inverse=True)
    expected = []
    for start in range(0, 17, 5):
        weights = np.zeros((min(5, 17 - start), 4), dtype=int)
        for indices in (np.array([0, 1]), np.array([2, 3])):
            weights[:, indices] = rng.multinomial(2, [0.5, 0.5], size=len(weights))
        for weight in weights:
            rows = np.repeat(np.arange(len(y)), weight[codes])
            expected.append(f1_score(y[rows], cand[rows], labels=[0, 1], average="macro", zero_division=0)
                            - f1_score(y[rows], ref[rows], labels=[0, 1], average="macro", zero_division=0))
    np.testing.assert_allclose(comparison.bootstrap_differences, expected, atol=1e-15, rtol=0)
    assert comparison.reference_score == pytest.approx(f1_score(y, ref, average="macro"))
    assert comparison.candidate_score == pytest.approx(f1_score(y, cand, average="macro"))
    assert comparison.lower_bound == pytest.approx(np.quantile(expected, 0.1))
    assert comparison.clusters_per_stratum == (2, 2)
    assert comparison.n_samples == 10 and comparison.n_clusters == 4


def test_identical_predictions_zero_difference_seed_and_exports(paired_data, tmp_path):
    y, ref, _, clusters = paired_data
    bootstrap = PairedClusterBootstrap(n_resamples=20)
    a = bootstrap.compare(y, ref, ref, clusters, strata=y)
    b = bootstrap.compare(y, ref, ref, clusters, strata=y)
    np.testing.assert_array_equal(a.bootstrap_differences, np.zeros(20))
    np.testing.assert_array_equal(a.bootstrap_differences, b.bootstrap_differences)
    assert a.observed_difference_pp == 0 and a.lower_bound_pp == 0
    assert not a.bootstrap_differences.flags.writeable
    directory = a.save(tmp_path)
    summary = json.loads((directory / "bootstrap_comparison.json").read_text())
    assert summary["difference_direction"] == "candidate minus reference"
    assert summary["n_resamples"] == 20
    values = np.loadtxt(directory / "bootstrap_differences.csv", delimiter=",", skiprows=1)
    assert values.shape == (20, 2)


def test_noninferiority_strict_boundary_and_percentage_point_units(paired_data):
    y, ref, cand, clusters = paired_data
    result = PairedClusterBootstrap(n_resamples=20).compare(y, ref, cand, clusters, strata=y)
    assert replace(result, lower_bound=-0.039).noninferiority(0.04).passed
    assert not replace(result, lower_bound=-0.04).noninferiority(0.04).passed
    assert not replace(result, lower_bound=-0.041).noninferiority(0.04).passed
    decision = result.noninferiority(0.04)
    assert decision.alpha == pytest.approx(0.1)
    for bad in (True, -0.1, 4, float("nan")):
        with pytest.raises(ValueError):
            result.noninferiority(bad)


def test_mixed_class_clusters_require_cluster_level_strata(paired_data):
    y, ref, cand, _ = paired_data
    mixed_clusters = np.arange(len(y)) % 2
    result = PairedClusterBootstrap(n_resamples=20).compare(y, ref, cand, mixed_clusters)
    assert not result.stratified
    with pytest.raises(ValueError, match="one stratum"):
        PairedClusterBootstrap(n_resamples=20).compare(y, ref, cand, mixed_clusters, strata=y)


@pytest.mark.parametrize("kwargs", [
    {"n_resamples": 1}, {"n_resamples": True}, {"batch_size": 0},
    {"confidence_level": 1}, {"confidence_level": float("nan")}, {"random_state": -1},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PairedClusterBootstrap(**kwargs)


def test_invalid_data_and_singleton_strata(paired_data):
    y, ref, cand, clusters = paired_data
    bootstrap = PairedClusterBootstrap(n_resamples=20)
    with pytest.raises(ValueError, match="one value per row"):
        bootstrap.compare(y, ref[:-1], cand, clusters)
    with pytest.raises(ValueError, match="two independent clusters"):
        bootstrap.compare(y, ref, cand, np.where(y == 0, "a", "b"), strata=y)
    with pytest.raises(ValueError, match="classes"):
        bootstrap.compare(y, ref, cand, clusters, classes=[0])
    with pytest.raises(ValueError, match="missing"):
        bootstrap.compare(y, ref, cand, np.array([None] + list(clusters[1:]), dtype=object))


def test_string_labels_and_explicit_absent_class(paired_data):
    y, ref, cand, clusters = paired_data
    encode = lambda values: np.where(values == 0, "negative", "positive")
    result = PairedClusterBootstrap(n_resamples=20).compare(
        encode(y), encode(ref), encode(cand), clusters, strata=y,
        classes=["negative", "positive", "unobserved"],
    )
    assert result.reference_score == pytest.approx(f1_score(
        encode(y), encode(ref), labels=["negative", "positive", "unobserved"],
        average="macro", zero_division=0,
    ))
