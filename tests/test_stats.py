import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from ab import stats as s


def test_srm_balanced_split_is_ok():
    assert s.srm_check(5000, 5050)["ok"]


def test_srm_detects_broken_split():
    assert not s.srm_check(5000, 5600)["ok"]


def test_ztest_matches_statsmodels():
    res = s.ztest_proportions(200, 4000, 250, 4000)
    z, p = proportions_ztest([250, 200], [4000, 4000])
    assert res["z"] == pytest.approx(z)
    assert res["p_value"] == pytest.approx(p)


def test_welch_ci_contains_true_diff():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 20, 5000)
    b = rng.normal(103, 20, 5000)
    res = s.welch_ttest(a, b)
    assert res["ci_low"] < 3 < res["ci_high"]
    assert res["p_value"] < 0.05


def test_cuped_keeps_mean_and_reduces_variance():
    rng = np.random.default_rng(2)
    x = rng.normal(50, 10, 10_000)
    y = x * 0.8 + rng.normal(0, 5, 10_000)
    y_c, theta = s.cuped(y, x)
    assert y_c.mean() == pytest.approx(y.mean())
    assert y_c.var() < y.var() * 0.5
    assert theta == pytest.approx(0.8, abs=0.05)


def test_sample_size_and_mde_are_consistent():
    n = s.sample_size_proportion(0.05, 0.005)
    assert s.mde_proportion(0.05, n) == pytest.approx(0.005, rel=0.05)


def test_holm_matches_statsmodels():
    p = [0.001, 0.02, 0.03, 0.2]
    expected = multipletests(p, method="holm")[0].tolist()
    assert s.holm(p) == expected


def test_bootstrap_ci_close_to_welch():
    rng = np.random.default_rng(3)
    a = rng.exponential(100, 4000)
    b = rng.exponential(110, 4000)
    boot = s.bootstrap_diff(a, b, n_boot=2000)
    welch = s.welch_ttest(a, b)
    assert boot["ci_low"] == pytest.approx(welch["ci_low"], abs=3)
    assert boot["ci_high"] == pytest.approx(welch["ci_high"], abs=3)
