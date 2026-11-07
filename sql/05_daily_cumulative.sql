-- Накопленная конверсия по дням попадания в тест — чтобы увидеть эффект новизны
-- и понять, стабилизировалась ли разница.
WITH daily AS (
    SELECT exposure_date,
           "group",
           COUNT(*)       AS users,
           SUM(converted) AS buyers
    FROM user_metrics
    GROUP BY exposure_date, "group"
)
SELECT exposure_date,
       "group",
       users,
       buyers,
       SUM(users)  OVER w AS cum_users,
       SUM(buyers) OVER w AS cum_buyers,
       ROUND(SUM(buyers) OVER w * 100.0 / SUM(users) OVER w, 3) AS cum_conversion_pct
FROM daily
WINDOW w AS (PARTITION BY "group" ORDER BY exposure_date
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
ORDER BY exposure_date, "group";
