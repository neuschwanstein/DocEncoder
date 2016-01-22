import psycopg2
from collections import namedtuple
from collections import Iterable
import urllib.request
from bs4 import BeautifulSoup
import dateutil.parser
import re

Article = namedtuple('Article', 'feedly_id location date title content')
Reuters = namedtuple('Reuters','feedly_id url')

missing_reuters_query = """
SELECT A.feedly_id,A.href FROM
reuters_usmarkets A 
LEFT JOIN
articles B
ON A.feedly_id = B.feedly_id
WHERE B.feedly_id IS NULL;
"""

insert_article_query = """
INSERT INTO articles
(feedly_id,location,date,title,content)
VALUES (%s,%s,%s,%s,%s)
"""

select_article_query = """
SELECT {} FROM articles LIMIT {}""".format(','.join(Article._fields), "{}")

delete_reuters_query = """
DELETE FROM reuters_usmarkets
WHERE feedly_id='{}'"""

conn = psycopg2.connect("dbname=TBM")
cur = conn.cursor()

class Fetch:
    def __init__(self, count="ALL"):
        batch_size = 25
        self.batch_size = batch_size
        cur.execute(select_article_query.format(count))
        self.records = []

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.records) == 0:
            self.records = cur.fetchmany(self.batch_size)
            if len(self.records) == 0:
                raise StopIteration
        return Article(*self.records.pop(0))

def fetch(count="ALL"):
    return Fetch(count)

def get_missing(count=100):
    cur.execute(missing_reuters_query)
    records = cur.fetchmany(count)
    return [Reuters._make(record) for record in records]

def insert(articles):
    if not isinstance(articles, Iterable):
        articles = [articles]
        
    for article in articles:
        cur.execute(insert_article_query,
                    (article.feedly_id, article.location,
                     article.date, article.title, article.content))
    conn.commit()

def delete(article):
    query = delete_reuters_query.format(article.feedly_id)
    cur.execute(query)
    conn.commit()

def parse(url):
    with urllib.request.urlopen(url) as response:
        html = response.read()
    
    soup = BeautifulSoup(html, "lxml")

    # Raw data
    title = soup.find("h1", {"class": "article-headline"}).text
    date = dateutil.parser.parse(
        soup.find("span", {"class": "timestamp"}).text)
    location = soup.find("span", {"class": "location"})

    # Manual cleanup
    location = location.text if location else ''

    pars = [p.text.replace('\n',' ') for p in
            soup.find(id="articleText").find_all("p")]

    # Remove early and trailing whitespace
    pars = [re.sub(r'^(\s*)(.+?)(\s*)$',r'\2',p) for p in pars if p]

    if pars[0].startswith(location + ' '):
        pars[0] = pars[0].replace(location + ' ','', 1) # at most once

    # Remove redundant date
    pars[0] = re.sub(r'\w{3} \d{2} ','',pars[0])

    last_par = pars[-1]
    if last_par[0] == '(' and last_par[-1] == ')':
        pars = pars[:-1]

    article_text = "\n\n".join(pars)

    return { 'location': location, 'date': date, 'title': title, 'content': article_text }

def add_missing(count=25):
    '''TODO: Add threaded process. Goes and parse missing Reuters articles, and then adds
    their content to the database.

    '''
    missings = get_missing(count)
    articles = []
    for i,missing in enumerate(missings):
        print("Parsing {} out of {}".format(i+1,count))
        try:
            article = parse(missing.url)
        except:
            delete(missing)
            print('Error! {}'.format(missing.url))
            continue
        article['feedly_id'] = missing.feedly_id
        article = Article(**article)
        articles.append(article)

        if i>0 and i%200 == 0:
            print("Inserting new articles...")
            insert(articles)
            articles = []
            print("Done.")

def close():
    conn.close()
