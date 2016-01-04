import psycopg2
from collections import namedtuple

Article = namedtuple('Article', 'id feedly_id authors published_date title article')

class ArticleReader:
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
