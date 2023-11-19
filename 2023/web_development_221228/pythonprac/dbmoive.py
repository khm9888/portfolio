from pymongo import MongoClient
client = MongoClient(
    'mongodb+srv://haemin:goals3068@cluster0.gntqb9x.mongodb.net/?retryWrites=true&w=majority')
db = client.dbsparta

movie = db.movies.find_one({"title": "가버나움"})
point = movie["point"]
print(point)

candidates = db.movies.find({"point": point})
for movie_one in candidates:
    print(movie_one["title"])

db.movies.update_one({"title": "가버나움"}, {'$set': {"point": "0"}})
