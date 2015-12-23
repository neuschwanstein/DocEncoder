from newspaper import Article
import psycopg2

QUERY_STRING = """
INSERT INTO articles_usmarkets
(authors,published_date,title,article)
VALUES (%s,%s,%s,%s)
"""

conn = psycopg2.connect("dbname=TBM")
cur = conn.cursor()
cur.execute("SELECT href from reuters_usmarkets")
records = cur.fetchmany(2)

for record in records:
    url = record[0]
    article = Article(url)
    article.download()
    article.parse()

    authors = "/".join(article.authors)
    published_date = article.publish_date
    title = article.title
    content = article.text

    # print((authors,published_date,title,content)) 

    cur.execute(QUERY_STRING, \
                (authors,published_date,title,content))

conn.commit()
print("DONE")

conn.close()
