import urllib.request
from datetime import datetime
import dateutil.parser
import re
from bs4 import BeautifulSoup

# url = "http://www.reuters.com/article/us-usa-stocks-idUSKBN0UM1C020160109"
url = "http://feeds.reuters.com/~r/news/usmarkets/~3/JHEednHzXJg/story01.htm"

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

pars[0] = re.sub(r'\w{3} \d{2} ','',pars[0])

last_par = pars[-1]
if last_par[0] == '(' and last_par[-1] == ')':
    pars = pars[:-1]

article_text = "\n\n".join(pars)

article = { 'location': location, 'title': title, 'content': article_text }
