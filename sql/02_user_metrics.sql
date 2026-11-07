-- Одна строка на пользователя: группа, сегменты и результат за время теста.
-- Заказы считаем только с даты попадания в тест; возвраты не учитываем в выручке.
WITH leaked AS (
    SELECT user_id
    FROM exposures
    GROUP BY user_id
    HAVING COUNT(DISTINCT "group") > 1
),
clean_users AS (
    SELECT e.*
    FROM exposures e
    WHERE e.user_id NOT IN (SELECT user_id FROM leaked)
),
user_orders AS (
    SELECT o.user_id,
           COUNT(*)                                            AS orders,
           SUM(CASE WHEN NOT o.refunded THEN o.revenue ELSE 0 END) AS revenue,
           SUM(CASE WHEN o.refunded THEN 1 ELSE 0 END)         AS refunds
    FROM orders o
    JOIN clean_users u USING (user_id)
    WHERE o.order_date >= u.exposure_date
    GROUP BY o.user_id
)
SELECT u.user_id,
       u."group",
       u.exposure_date,
       u.device,
       u.user_type,
       u.pre_revenue,
       COALESCE(uo.orders, 0)          AS orders,
       (COALESCE(uo.orders, 0) > 0)::INT AS converted,
       COALESCE(uo.revenue, 0)         AS revenue,
       COALESCE(uo.refunds, 0)         AS refunds
FROM clean_users u
LEFT JOIN user_orders uo USING (user_id);
