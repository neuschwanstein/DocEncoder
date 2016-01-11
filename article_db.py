import psycopg2
from collections import namedtuple
# from contextlib import contextmanager
# @contextmanager
# def 

Article = namedtuple('Article', 'id feedly_id location date title content')
Reuters = namedtuple('Reuters','feedly_id href')

class Fetch:
    def __init__(self, count=0):
        if count == 0:
            count = "ALL"
        batch_size = 25
        self.batch_size = batch_size
        self.conn = psycopg2.connect("dbname=TBM")
        self.cur = self.conn.cursor()
        self.cur.execute("SELECT * FROM articles_usmarkets LIMIT {}".format(count))
        self.records = []

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.records) == 0:
            self.records = self.cur.fetchmany(self.batch_size)
            if len(self.records) == 0:
                raise StopIteration
        return Article(*self.records.pop(0))

    def __del__(self):
        self.conn.close()

def fetch(count=0):
    return Fetch(count)

def missing_urls(count=100):
    missing_reuters_query = """
    SELECT A.feedly_id,A.href FROM
    reuters_usmarkets A 
    LEFT JOIN
    articles_usmarkets B
    ON A.feedly_id = B.feedly_id
    WHERE B.feedly_id IS NULL;
    """
    conn = psycopg2.connect("TBM")
    cur = conn.cursor()
    cur.execute(missing_reuters_query)
    records = cur.fetchmany(count)
    conn.close()

def insert(article):
    insert_article_query = """
    INSERT INTO articles_usmarkets
    (feedly_id,authors,published_date,title,article)
    VALUES (%s,%s,%s,%s,%s)
    """
    conn = psycopg2.connect("dbname=TBM")
    cur = conn.cursor()
    cur.execute(insert_article_query, \
                (article.feedly_id, article.location, article.date, article.title, article.content))
    conn.commit()
    conn.close()
