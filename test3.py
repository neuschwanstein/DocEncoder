import urllib.request
from datetime import datetime
import dateutil.parser
from bs4 import BeautifulSoup

# url = "http://www.reuters.com/article/us-eurozone-economy-dawn-analysis-idUSKCN0UO0DQ20160110?feedType=RSS&feedName=businessNews"
url = "http://www.reuters.com/article/china-orient-ipo-idUSL3N14C04620151223?feedType=RSS&feedName=marketsNews"

with urllib.request.urlopen(url) as response:
    html = response.read()

soup = BeautifulSoup(html, "lxml")

#Raw data
article_title = soup.find("h1", {"class": "article-headline"}).text
article_date = dateutil.parser.parse(
    soup.find("span", {"class": "timestamp"}).text)
article_location = soup.find("span", {"class": "location"}).text
article_text = soup.find(id="articleText").text

