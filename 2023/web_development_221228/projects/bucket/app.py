from pymongo import MongoClient
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
client = MongoClient(
    'mongodb+srv://haemin:goals3068@cluster0.gntqb9x.mongodb.net/?retryWrites=true&w=majority')
db = client.dbsparta


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/bucket", methods=["POST"])
def bucket_post():
    bucket_receive = request.form['bucket_give']
    print(bucket_receive)

    bucket_list = list(db.bucket.find({}, {"_id": False}))
    bucket_cnt = len(bucket_list)
    doc = {
        "num": bucket_cnt+1,
        "bucket": bucket_receive,
        "done": 0

    }
    db.bucket.insert_one(doc)
    return jsonify({'msg': '등록 완료!'})


@app.route("/bucket/done", methods=["POST"])
def bucket_done():
    num_receive = request.form['num_give']
    db.bucket.update_one({"num": int(num_receive)}, {"$set": {"done": 1}})
    return jsonify({'msg': '버킷 완료!'})


@app.route("/bucket", methods=["GET"])
def bucket_get():
    bucket_list = list(db.bucket.find({}, {"_id": False}))
    return jsonify({'buckets': bucket_list})


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
