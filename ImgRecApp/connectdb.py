import pymongo
# Connecting to the database and the collection
def connectDb():
    client = pymongo.MongoClient("mongodb+srv://arman:DNf7gZN22RQgyzpj@cluster0.vjnupxv.mongodb.net/?retryWrites=true&w=majority")
    db = client["imgrecapp"]
    collection = db["nutritions"]
    return collection
