-- Конверсия по сегментам: устройство × тип пользователя.
SELECT device,
       user_type,
       "group",
       COUNT(*)                       AS users,
       SUM(converted)                 AS buyers,
       ROUND(AVG(converted) * 100, 2) AS conversion_pct
FROM user_metrics
GROUP BY device, user_type, "group"
ORDER BY device, user_type, "group";
