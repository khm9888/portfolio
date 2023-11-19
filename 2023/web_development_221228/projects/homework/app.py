from pymongo import MongoClient
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
client = MongoClient(
    'mongodb+srv://haemin:goals3068@cluster0.gntqb9x.mongodb.net/?retryWrites=true&w=majority')
db = client.dbsparta


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/homework", methods=["POST"])
def homework_post():
    nickname_receive = request.form['nickname_give']
    cheerupment_receive = request.form['cheerupment_give']
    print(nickname_receive, cheerupment_receive)
    db.homeworks.insert_one({"name": nickname_receive,
                             "comment": cheerupment_receive})
    return jsonify({'msg': 'POST 연결 완료!'})


@app.route("/homework", methods=["GET"])
def homework_get():
    ment_list = list(db.homeworks.find({}, {"_id": False}))
    # print(ment_list)
    return jsonify({'ment_list': ment_list})


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
