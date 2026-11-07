"""Генерация синтетических данных A/B-теста нового чекаута.

Сценарий: интернет-магазин тестирует одностраничное оформление заказа (группа B)
против старого трёхшагового (группа A). Тест шёл 14 дней.

Данные синтетические, но с «реальными» проблемами, которые аналитик должен найти:
- часть пользователей попала в обе группы (утечка сплита);
- выручка с тяжёлым хвостом (несколько очень крупных заказов);
- эффект различается по сегментам (мобильные vs десктоп);
- в первые дни заметен эффект новизны.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_USERS = 80_000
START = pd.Timestamp("2026-09-01")
DAYS = 14
LEAKED_USERS = 240


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    user_id = np.arange(100_001, 100_001 + N_USERS)
    group = rng.choice(["A", "B"], size=N_USERS)
    day = rng.integers(0, DAYS, size=N_USERS)
    device = rng.choice(["mobile", "desktop"], size=N_USERS, p=[0.6, 0.4])
    user_type = rng.choice(["new", "returning"], size=N_USERS, p=[0.45, 0.55])

    # Выручка пользователя за 30 дней ДО теста (нужна для CUPED)
    returning = user_type == "returning"
    bought_before = returning & (rng.random(N_USERS) < 0.5)
    pre_revenue = np.where(
        bought_before, np.round(rng.lognormal(8.0, 0.8, N_USERS), 0), 0.0
    )
    # «Склонность к покупке» — общая для периода до и во время теста
    propensity = np.where(bought_before, 1.0 + (pre_revenue / 3000) ** 0.9, 1.0)

    base = np.where(device == "mobile", 0.030, 0.048)
    base = base + np.where(returning, 0.012, 0.0)
    p = base * propensity

    lift = np.where(device == "mobile", 1.17, 1.01)
    novelty = np.where(day < 3, 1.06, 1.0)
    p = np.where(group == "B", p * lift * novelty, p)

    converted = rng.random(N_USERS) < p

    users = pd.DataFrame(
        {
            "user_id": user_id,
            "group": group,
            "exposure_date": (START + pd.to_timedelta(day, unit="D")).date,
            "device": device,
            "user_type": user_type,
            "pre_revenue": pre_revenue,
        }
    )

    # Заказы
    rows = []
    order_id = 5_000_001
    for uid, g, d, conv, prop, pre in zip(user_id, group, day, converted, propensity, pre_revenue):
        if not conv:
            continue
        n_orders = 1 + (rng.random() < 0.12)
        for k in range(n_orders):
            check = rng.lognormal(8.05, 0.65) * (0.97 if g == "B" else 1.0)
            # постоянные клиенты с большими тратами до теста и сейчас тратят больше
            check *= (pre / 3000) ** 0.8 if pre > 0 else 0.8
            if rng.random() < 0.004:  # редкие оптовые заказы — тяжёлый хвост
                check *= rng.uniform(8, 20)
            order_day = min(d + k * int(rng.integers(1, 4)), DAYS - 1)
            rows.append(
                (
                    order_id,
                    uid,
                    (START + pd.Timedelta(days=int(order_day))).date(),
                    round(check, 0),
                    bool(rng.random() < 0.06),
                )
            )
            order_id += 1
    orders = pd.DataFrame(
        rows, columns=["order_id", "user_id", "order_date", "revenue", "refunded"]
    )

    # Утечка сплита: часть пользователей увидела и вторую версию
    leaked = rng.choice(users.index, size=LEAKED_USERS, replace=False)
    dup = users.loc[leaked].copy()
    dup["group"] = np.where(dup["group"] == "A", "B", "A")
    exposures = pd.concat([users, dup]).sort_values(["exposure_date", "user_id"])

    return exposures.reset_index(drop=True), orders


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    exposures, orders = generate()
    exposures.to_csv(out / "exposures.csv", index=False)
    orders.to_csv(out / "orders.csv", index=False)
    print(f"exposures: {len(exposures):,} строк, orders: {len(orders):,} строк")


if __name__ == "__main__":
    main()
