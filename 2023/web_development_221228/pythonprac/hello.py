from bs4 import BeautifulSoup
import requests
from pymongo import MongoClient
client = MongoClient(
    'mongodb+srv://haemin:goals3068@cluster0.gntqb9x.mongodb.net/?retryWrites=true&w=majority')
db = client.dbsparta

# 타겟 URL을 읽어서 HTML를 받아오고,
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
data = requests.get(
    'https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=pnt&date=20210829', headers=headers)

# HTML을 BeautifulSoup이라는 라이브러리를 활용해 검색하기 용이한 상태로 만듦
# soup이라는 변수에 "파싱 용이해진 html"이 담긴 상태가 됨
# 이제 코딩을 통해 필요한 부분을 추출하면 된다.
soup = BeautifulSoup(data.text, 'html.parser')

# title = soup.select_one(
#     "#old_content > table > tbody > tr:nth-child(2) > td.title > div > a")


movies = soup.select("#old_content > table > tbody > tr")

for movie in movies:
    a = movie.select_one("td.title > div > a")
    if a != None:
        rank = movie.select_one("td:nth-child(1) > img")["alt"]
        # #old_content > table > tbody > tr:nth-child(13) > td:nth-child(1) > img
        title = a.text
        point = movie.select_one("td.point").text
        # #old_content > table > tbody > tr:nth-child(2) > td.point
        print(rank, title, point)
        doc = {'rank': rank,
               'title': title,
               'point': point
               }
        db.movies.insert_one(doc)

# print(movies[0:2])
# print(len(movies))
#############################
# (입맛에 맞게 코딩)
#############################
