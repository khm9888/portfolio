from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
client = MongoClient(
    'mongodb+srv://haemin:goals3068@cluster0.gntqb9x.mongodb.net/?retryWrites=true&w=majority')
db = client.dbsparta


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/movie", methods=["POST"])
def movie_post():
    # 사용자로 부터 url 등 정보 가져오기
    url_receive = request.form['url_give']
    star_receive = request.form['star_give']
    comment_receive = request.form['comment_give']

    # 가져온 url로 크롤링 진행

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
    data = requests.get(url_receive, headers=headers)

    soup = BeautifulSoup(data.text, 'html.parser')

    # 크롤링한 데이터를 DB 넣기

    # 여기에 코딩을 해서 meta tag를 먼저 가져와보겠습니다.

    # print(soup)

    title = soup.select_one("meta[property='og:title']")["content"]
    image = soup.select_one("meta[property='og:image']")["content"]
    description = soup.select_one("meta[property='og:description']")["content"]
    # title = soup.select_one('meta[property="og:title"]')

    doc = {
        "title": title,
        "image": image,
        "description": description,
        "star": star_receive,
        "comment": comment_receive,
    }
    db.movies.insert_one(doc)

    return jsonify({'msg': '저장 완료!'})


@app.route("/movie", methods=["GET"])
def movie_get():
    movies = list(db.movies.find({}, {"_id": False}))
    # for movie in movies:
    return jsonify({'movies': movies})


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
