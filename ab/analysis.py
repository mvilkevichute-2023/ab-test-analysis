"""Полный анализ эксперимента: от проверки сплита до решения.

Запуск:  python -m ab.analysis
Печатает отчёт в консоль и сохраняет графики в images/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from ab import stats as s
from ab.data import connect, query

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"

COLOR_A = "#3F7FBF"  # контроль
COLOR_B = "#C8327A"  # тест
INK = "#2B2B2B"
MUTED = "#6B6B6B"
GRID = "#E6E6E6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

WINSOR_Q = 0.995


def load() -> tuple[pd.DataFrame, dict]:
    con = connect()
    leaked = query(con, "01_split_leak.sql")
    um = con.execute("SELECT * FROM user_metrics ORDER BY user_id").df()
    info = {
        "con": con,
        "exposures": con.execute("SELECT COUNT(*) FROM exposures").fetchone()[0],
        "leaked": len(leaked),
    }
    return um, info


def run(um: pd.DataFrame) -> dict:
    a, b = um[um.group == "A"], um[um.group == "B"]
    res: dict = {}

    res["srm"] = s.srm_check(len(a), len(b))
    res["conversion"] = s.ztest_proportions(a.converted.sum(), len(a), b.converted.sum(), len(b))
    res["mde"] = s.mde_proportion(res["conversion"]["p_a"], len(a))

    res["arpu_raw"] = s.welch_ttest(a.revenue, b.revenue)
    cap = um.revenue.quantile(WINSOR_Q)
    um = um.assign(revenue_w=um.revenue.clip(upper=cap))
    res["winsor_cap"] = cap
    res["arpu_winsor"] = s.welch_ttest(um[um.group == "A"].revenue_w, um[um.group == "B"].revenue_w)
    res["arpu_bootstrap"] = s.bootstrap_diff(
        um[um.group == "A"].revenue_w.values, um[um.group == "B"].revenue_w.values
    )

    y_c, theta = s.cuped(um.revenue_w.values, um.pre_revenue.values)
    um = um.assign(revenue_cuped=y_c)
    res["cuped_theta"] = theta
    res["cuped_var_reduction"] = 1 - y_c.var() / um.revenue_w.var()
    res["arpu_cuped"] = s.welch_ttest(um[um.group == "A"].revenue_cuped, um[um.group == "B"].revenue_cuped)

    # Защитные метрики: средний чек и доля возвратов
    res["aov"] = {
        g: d.revenue.sum() / (d.orders - d.refunds).sum() for g, d in um.groupby("group")
    }
    res["refunds"] = s.ztest_proportions(a.refunds.sum(), a.orders.sum(), b.refunds.sum(), b.orders.sum())

    # Сегменты + поправка Холма
    seg_rows = []
    for (device, utype), g in um.groupby(["device", "user_type"]):
        ga, gb = g[g.group == "A"], g[g.group == "B"]
        z = s.ztest_proportions(ga.converted.sum(), len(ga), gb.converted.sum(), len(gb))
        seg_rows.append(
            {
                "segment": f"{'Мобильные' if device == 'mobile' else 'Десктоп'} · "
                f"{'новые' if utype == 'new' else 'постоянные'}",
                "users": len(g),
                "conv_a": z["p_a"],
                "conv_b": z["p_b"],
                "uplift": z["uplift"],
                "ci_low": z["uplift_ci_low"],
                "ci_high": z["uplift_ci_high"],
                "p_value": z["p_value"],
            }
        )
    seg = pd.DataFrame(seg_rows)
    seg["significant_holm"] = s.holm(seg.p_value.tolist())
    res["segments"] = seg

    dev_rows = []
    for device, g in um.groupby("device"):
        ga, gb = g[g.group == "A"], g[g.group == "B"]
        z = s.ztest_proportions(ga.converted.sum(), len(ga), gb.converted.sum(), len(gb))
        dev_rows.append({"device": device, **{k: z[k] for k in ("p_a", "p_b", "uplift", "uplift_ci_low", "uplift_ci_high", "p_value")}})
    res["devices"] = pd.DataFrame(dev_rows)
    res["um"] = um
    return res


# ---------- графики ----------

def _fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def plot_conversion(res: dict, path: Path) -> None:
    c = res["conversion"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (g, p, color) in enumerate([("A · старый чекаут", c["p_a"], COLOR_A), ("B · новый чекаут", c["p_b"], COLOR_B)]):
        n = res["um"].group.value_counts()["A" if i == 0 else "B"]
        se = np.sqrt(p * (1 - p) / n)
        ax.bar(i, p, width=0.5, color=color)
        ax.errorbar(i, p, yerr=1.96 * se, color=INK, capsize=6, linewidth=1.5)
        ax.text(i, p + 1.96 * se + 0.0015, _fmt_pct(p, 2), ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1], ["A · старый чекаут", "B · новый чекаут"])
    ax.set_ylim(0, max(c["p_a"], c["p_b"]) * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: _fmt_pct(v, 0))
    ax.grid(axis="x", visible=False)
    ax.set_title(
        f"Конверсия в покупку: +{_fmt_pct(c['uplift'])} ("
        + ("p < 0,001" if c["p_value"] < 0.001 else f"p = {c['p_value']:.3f}".replace(".", ","))
        + ")",
        loc="left", color=INK, fontsize=13, fontweight="bold",
    )
    ax.text(0, -0.2, "Чёрные отрезки — 95% доверительный интервал", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_cumulative(daily: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for g, color, label in [("A", COLOR_A, "A · старый"), ("B", COLOR_B, "B · новый")]:
        d = daily[daily.group == g]
        ax.plot(pd.to_datetime(d.exposure_date), d.cum_conversion_pct / 100, color=color, linewidth=2, marker="o", markersize=4)
        last = d.iloc[-1]
        ax.text(pd.to_datetime(last.exposure_date) + pd.Timedelta(hours=10), last.cum_conversion_pct / 100, label,
                color=INK, va="center", fontsize=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(lambda v, _: _fmt_pct(v, 1))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m"))
    ax.set_title("Накопленная конверсия: в первые дни эффект новизны,\nпотом разница сужается, но остаётся",
                 loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.set_xlim(right=pd.to_datetime(daily.exposure_date.max()) + pd.Timedelta(days=2))
    fig.savefig(path)
    plt.close(fig)


def plot_segments(seg: pd.DataFrame, path: Path) -> None:
    seg = seg.sort_values("uplift")
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(seg))
    colors = [COLOR_B if sig else "#B9B9B9" for sig in seg.significant_holm]
    ax.hlines(y, seg.ci_low, seg.ci_high, color=colors, linewidth=2.5)
    ax.scatter(seg.uplift, y, color=colors, s=70, zorder=3, edgecolor="white", linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1)
    for yi, (_, r) in zip(y, seg.iterrows()):
        ax.text(r.ci_high + 0.015, yi, f"{'+' if r.uplift >= 0 else ''}{_fmt_pct(r.uplift)}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, seg.segment)
    ax.xaxis.set_major_formatter(lambda v, _: _fmt_pct(v, 0))
    ax.grid(axis="y", visible=False)
    ax.set_title("Эффект на конверсию по сегментам (95% ДИ)", loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.text(0, -0.2, "Розовым — значимо после поправки Холма, серым — нет", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_bootstrap(res: dict, path: Path) -> None:
    bt = res["arpu_bootstrap"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.hist(bt["diffs"], bins=60, color="#E7A9C7", edgecolor="white", linewidth=0.5)
    ax.axvspan(bt["ci_low"], bt["ci_high"], color=COLOR_B, alpha=0.08)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.axvline(bt["diff"], color=COLOR_B, linewidth=2)
    ax.text(bt["diff"] + 1, ax.get_ylim()[1] * 0.97, f"+{bt['diff']:.1f} ₽ на пользователя".replace(".", ","),
            color=INK, fontsize=10, va="top", bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.text(0, -0.24, f"Закрашено — 95% ДИ: от +{bt['ci_low']:.1f} до +{bt['ci_high']:.1f} ₽; ноль в него не входит".replace(".", ","),
            transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_title("Бутстрап разницы выручки на пользователя (B − A)", loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.set_xlabel("₽ на пользователя")
    fig.savefig(path)
    plt.close(fig)


def main() -> dict:
    IMG.mkdir(exist_ok=True)
    um, info = load()
    res = run(um)
    daily = query(info["con"], "05_daily_cumulative.sql")
    summary = query(info["con"], "03_group_summary.sql")

    plot_conversion(res, IMG / "01_conversion.png")
    plot_cumulative(daily, IMG / "02_cumulative.png")
    plot_segments(res["segments"], IMG / "03_segments.png")
    plot_bootstrap(res, IMG / "04_bootstrap_arpu.png")

    c = res["conversion"]
    print("=" * 60)
    print(f"Показов в логах: {info['exposures']:,}; попали в обе группы: {info['leaked']} — исключены")
    print(summary.to_string(index=False))
    print(f"\nSRM: доля A = {res['srm']['share_a']:.4f}, p = {res['srm']['p_value']:.3f} -> {'OK' if res['srm']['ok'] else 'СПЛИТ СЛОМАН'}")
    print(f"MDE при такой выборке: {res['mde']*100:.2f} п.п. ({res['mde']/c['p_a']*100:.1f}% относительно)")
    print(f"\nКонверсия: {c['p_a']*100:.2f}% -> {c['p_b']*100:.2f}%  uplift {c['uplift']*100:+.1f}% "
          f"[{c['uplift_ci_low']*100:+.1f}%; {c['uplift_ci_high']*100:+.1f}%], p = {c['p_value']:.5f}")
    for key, name in [("arpu_raw", "ARPU сырой"), ("arpu_winsor", f"ARPU винзор. ({WINSOR_Q:.1%})"), ("arpu_cuped", "ARPU CUPED")]:
        r = res[key]
        print(f"{name:<24} {r['mean_a']:.1f} -> {r['mean_b']:.1f} ₽, {r['uplift']*100:+.1f}% "
              f"ДИ разницы [{r['ci_low']:+.1f}; {r['ci_high']:+.1f}] ₽, p = {r['p_value']:.4f}")
    bt = res["arpu_bootstrap"]
    print(f"Бутстрап ARPU (винзор.): ДИ [{bt['ci_low']:+.1f}; {bt['ci_high']:+.1f}] ₽, p = {bt['p_value']:.4f}")
    print(f"Порог винзоризации: {res['winsor_cap']:,.0f} ₽; CUPED theta = {res['cuped_theta']:.4f}, "
          f"снижение дисперсии {res['cuped_var_reduction']*100:.1f}%")
    print(f"\nСредний чек: A {res['aov']['A']:,.0f} ₽, B {res['aov']['B']:,.0f} ₽")
    r = res["refunds"]
    print(f"Возвраты: {r['p_a']*100:.2f}% -> {r['p_b']*100:.2f}%, p = {r['p_value']:.3f}")
    print("\nПо устройствам:")
    print(res["devices"].round(4).to_string(index=False))
    print("\nПо сегментам:")
    print(res["segments"].round(4).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
