from collections import namedtuple
from newspaper import Article
import psycopg2

QUERY_STRING = """
INSERT INTO articles_usmarkets
(feedly_id,authors,published_date,title,article)
VALUES (%s,%s,%s,%s,%s)
"""

GET_MISSING_ARTICLES = """
SELECT A.feedly_id,A.href FROM
reuters_usmarkets A 
LEFT JOIN
articles_usmarkets B
ON A.feedly_id = B.feedly_id
WHERE B.feedly_id IS NULL;
"""

Reuters = namedtuple('Reuters','feedly_id href')

conn = psycopg2.connect("dbname=TBM")
cur = conn.cursor()
cur.execute(GET_MISSING_ARTICLES)

records = cur.fetchmany(793)

# TODO
# Learning is not optimal:
# 1. City reporting is in <span class="article">SHANGHAI</class>
# 2. Date is not complete when converted to db form (psycopg2 side)
# 3. Authors are not correctly reported 
for record in records:          # TODO Can be made parallel.
    record = Reuters(*record)
    article = Article(record.href)
    
    article.download()
    article.parse()

    authors = "/".join(article.authors)
    published_date = article.publish_date
    title = article.title
    content = article.text

    cur.execute(QUERY_STRING, \
                (record.feedly_id,authors,published_date,title,content))

conn.commit()
print("DONE")

conn.close()
