SELECT A.feedly_id,A.href FROM
reuters_usmarkets A 
LEFT JOIN
articles_usmarkets B
ON A.feedly_id = B.feedly_id
WHERE B.feedly_id IS NULL;