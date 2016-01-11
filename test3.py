import urllib.request
from datetime import datetime
import dateutil.parser
import re
from bs4 import BeautifulSoup

url = "http://www.reuters.com/article/us-usa-stocks-idUSKBN0UM1C020160109"

with urllib.request.urlopen(url) as response:
    html = response.read()

soup = BeautifulSoup(html, "lxml")

# Raw data
article_title = soup.find("h1", {"class": "article-headline"}).text
article_date = dateutil.parser.parse(
    soup.find("span", {"class": "timestamp"}).text)
article_location = soup.find("span", {"class": "location"})


# Manual cleanup
article_location = article_location.text if article_location else ''

article_paragraphs = [p.text.replace('\n',' ') for p in
                      soup.find(id="articleText").find_all("p")]

# Remove early and trailing whitespace
article_paragraphs = [re.sub(r'^(\s*)(.+?)(\s*)$',r'\2',p) for p in article_paragraphs if p]

if article_paragraphs[0].startswith(article_location + ' '):
    article_paragraphs[0] = article_paragraphs[0].replace(article_location + ' ','', 1) # at most once

article_paragraphs[0] = re.sub(r'\w{3} \d{2} ','',article_paragraphs[0])

last_paragraph = article_paragraphs[-1]
if last_paragraph[0] == '(' and last_paragraph[-1] == ')':
    article_paragraphs = article_paragraphs[:-1]

article_text = "\n\n".join(article_paragraphs)
