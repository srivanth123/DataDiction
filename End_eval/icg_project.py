import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://raw.githubusercontent.com/PIYUSH06VERMA/DataDiction/refs/heads/main/End_eval/dataset.html"

response = requests.get(url)
html = response.text

soup = BeautifulSoup(html, "html.parser")

table = soup.find("table")

headers = [th.text.strip() for th in table.find_all("th")]

rows = []
for tr in table.find_all("tr")[1:]:
    rows.append([td.text.strip() for td in tr.find_all("td")])

df = pd.DataFrame(rows, columns=headers)

print(df.head())
