import requests
URL = "http://127.0.0.1:5000/predict"
requests.get("http://127.0.0.1:5000/")
response = requests.post(URL, json='{"text":"Грубые сотрудники, навязали кредит, долго обсуживали!"}')
print("Текст: Грубые сотрудники, навязали кредит, долго обсуживали! Результат: ", response.json()["answer"][1:-1])
response = requests.post(URL, json='{"text":"Вежливые сотрудники, быстрое обслуживание"}')
print("Текст: Вежливые сотрудники, быстрое обслуживание! Результат: ", response.json()["answer"][1:-1])
