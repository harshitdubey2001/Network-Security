from pymongo import MongoClient

uri = "mongodb+srv://harshitdubey7896_db_user:Admin123@cluster0.azxb6ub.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("Connected to MongoDB Atlas!")
except Exception as e:
    print("Connection failed:", e)
