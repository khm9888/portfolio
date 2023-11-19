from bs4 import BeautifulSoup
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
data = requests.get(
    'https://www.genie.co.kr/chart/top200?ditc=M&rtm=N&ymd=20210701', headers=headers)

# HTML을 BeautifulSoup이라는 라이브러리를 활용해 검색하기 용이한 상태로 만듦
# soup이라는 변수에 "파싱 용이해진 html"이 담긴 상태가 됨
# 이제 코딩을 통해 필요한 부분을 추출하면 된다.
soup = BeautifulSoup(data.text, 'html.parser')
# print(soup)
songs = soup.select(
    "#body-content > div.newest-list > div > table > tbody > tr")
# print(songs)
print(len(songs))

for song in songs:
    rank = song.select_one(
        " td.number").text[:2].strip()
    title = song.select_one(
        " td.info > a.title.ellipsis").text.strip()
    if title[:3] == "19금":
        title = title[3:].strip()
    artist = song.select_one(
        " td.info > a.artist.ellipsis").text.strip()

    # print(len(rank), len(title), len(artist))
    print(rank, title, artist)

# body-content > div.newest-list > div > table > tbody > tr:nth-child(2) > td.info > a.title.ellipsis
