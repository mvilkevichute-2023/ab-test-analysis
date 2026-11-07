"""Статистические функции для анализа A/B-теста.

Каждая функция возвращает словарь с понятными полями, чтобы результат было
удобно показывать в таблице или в тексте отчёта.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> dict:
    """Sample Ratio Mismatch: совпадает ли реальный сплит с запланированным.

    Если p-value < 0.001 — сплит сломан и результатам теста доверять нельзя.
    """
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    chi2, p = stats.chisquare([n_a, n_b], f_exp=expected)
    return {"share_a": n_a / total, "chi2": chi2, "p_value": p, "ok": p >= 0.001}


def ztest_proportions(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Z-тест для разницы конверсий + доверительный интервал разницы и относительного эффекта."""
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return {
        "p_a": p_a,
        "p_b": p_b,
        "diff": diff,
        "ci_low": diff - z_crit * se_diff,
        "ci_high": diff + z_crit * se_diff,
        "uplift": diff / p_a,
        "uplift_ci_low": (diff - z_crit * se_diff) / p_a,
        "uplift_ci_high": (diff + z_crit * se_diff) / p_a,
        "z": z,
        "p_value": p_value,
    }


def welch_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """T-тест Уэлча для средних (например, выручки на пользователя) с ДИ разницы."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(b, a, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    diff = b.mean() - a.mean()
    return {
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "diff": diff,
        "ci_low": diff - t_crit * se,
        "ci_high": diff + t_crit * se,
        "uplift": diff / a.mean(),
        "t": t,
        "p_value": p,
    }


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5_000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Бутстрап-ДИ для разницы средних — не требует нормальности, устойчив к выбросам в выборке."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = b[idx_b].mean(axis=1) - a[idx_a].mean(axis=1)
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # двусторонний p-value: доля бутстрап-разниц «по другую сторону нуля»
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": b.mean() - a.mean(), "ci_low": low, "ci_high": high, "p_value": min(p, 1.0), "diffs": diffs}


def cuped(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: снижает дисперсию метрики, используя ту же метрику до теста (ковариату x).

    y_cuped = y - theta * (x - mean(x)),  theta = cov(x, y) / var(x).
    Среднее метрики не меняется, а дисперсия падает на долю corr(x, y)^2 —
    тест становится чувствительнее без увеличения выборки.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(x, y)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), theta


def mde_proportion(p_base: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Минимальный абсолютный эффект в конверсии, который тест может надёжно заметить."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * np.sqrt(2 * p_base * (1 - p_base) / n_per_group)


def sample_size_proportion(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Сколько пользователей нужно в каждой группе, чтобы заметить эффект mde_abs."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_new = p_base + mde_abs
    var = p_base * (1 - p_base) + p_new * (1 - p_new)
    return int(np.ceil((z_a + z_b) ** 2 * var / mde_abs**2))


def holm(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Поправка Холма на множественные сравнения: какие гипотезы отвергаем."""
    order = np.argsort(p_values)
    m = len(p_values)
    reject = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject
