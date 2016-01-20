import psycopg2

Representation = namedtuple('Representation','article_id content')

insert_representations_query = '''
INSERT INTO reprensentations
(article_id content)
VALUES (%s,%s)
'''

conn = psycopg2.connect('dbname=TBM')
cur = conn.cursor()

def insert(representations):
    if not isinstance(representations,Iterable):
        representations = [representations]
    for i,rep in enumerate(representations):
        cur.execute(insert_representations_query,
                    (rep.article_id,rep.content))
    conn.commit()

