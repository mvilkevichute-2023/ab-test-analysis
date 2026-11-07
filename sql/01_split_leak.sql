-- Пользователи, которые попали в обе группы (утечка сплита).
-- Их нельзя честно отнести ни к A, ни к B — исключаем из анализа.
SELECT user_id,
       COUNT(DISTINCT "group") AS groups_seen
FROM exposures
GROUP BY user_id
HAVING COUNT(DISTINCT "group") > 1;
