import requests
from urllib.parse import urlparse
import psycopg2
from datetime import datetime

from RobustDictionary import RobustDictionary

FEEDLY_URL = "http://cloud.feedly.com/v3/streams/contents"
REUTERS_URL = "feed/http://feeds.reuters.com/news/usmarkets"
COUNT = 10000

conn = psycopg2.connect("dbname=TBM")
db = conn.cursor()

QUERY_STRING = """
INSERT INTO reuters_usmarkets
(feedly_id,href,published_date,title,engagement,engagement_rate,content)
VALUES (%s,%s,%s,%s,%s,%s,%s)
"""

continuation =  ''
while continuation is not None:
    r = requests.get(FEEDLY_URL, \
                     params = { "streamId": REUTERS_URL, \
                                "count": COUNT, \
                                "continuation": continuation })

    if not r.ok:
        raise Exception("Error code : " + str(r.status_code))

    items = RobustDictionary(r.json())
    continuation = items['continuation']

    for item in items['items']:
        item = RobustDictionary(item)

        feedly_id = item['id']
        published_date = datetime.fromtimestamp(int(item['published'])/1000)
        title = item['title']
        engagement = int(item['engagement']) if item['engagement'] is not None else 0
        engagement_rate = float(item['engagement_rate']) if item['engagement_rate'] is not None else 0
        href = item['alternate'][0]['href']

        if item['summary'] is None:
            content = None
        else:
            content = item['summary']['content']

        db.execute(QUERY_STRING, \
                   (feedly_id, href, published_date, title, engagement, engagement_rate, content))
    
    conn.commit()
    print("DONE")

conn.close()
