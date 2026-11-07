-- Сводка по группам: основные и защитные метрики.
SELECT "group",
       COUNT(*)                                   AS users,
       SUM(converted)                             AS buyers,
       ROUND(AVG(converted) * 100, 2)             AS conversion_pct,
       ROUND(SUM(revenue), 0)                     AS revenue,
       ROUND(AVG(revenue), 1)                     AS arpu,
       ROUND(SUM(revenue) / NULLIF(SUM(orders) - SUM(refunds), 0), 0) AS aov,
       ROUND(SUM(refunds) * 100.0 / NULLIF(SUM(orders), 0), 2)       AS refund_rate_pct
FROM user_metrics
GROUP BY "group"
ORDER BY "group";
