import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

load_dotenv()  # загрузка переменных из .env

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True,  # для отладки
    tlsCAFile=certifi.where()
)
db = client["word_helper_db"]  # имя базы данных
dictionary_collection = db["dictionary"]  # коллекция для словаря

# Проверяем, пуста ли коллекция
if dictionary_collection.count_documents({}) == 0:
    with open("cleaned_filtered_russian_words.json", "r", encoding="utf-8") as f:
        words = json.load(f)
    documents = [{"word": w} for w in words]
    result = dictionary_collection.insert_many(documents)
    print(f"Вставлено {len(result.inserted_ids)} слов в базу!")
else:
    print("Коллекция уже содержит данные. Импорт не требуется.")