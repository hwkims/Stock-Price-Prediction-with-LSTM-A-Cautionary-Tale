import requests

# 삼성전자 주식 데이터 조회 URL
url = 'https://api.koreainvestment.com/v1/stock/quote?code=005930'

# 요청
response = requests.get(url)

# 데이터 출력
if response.status_code == 200:
    data = response.json()
    print("주식 데이터:", data)
else:
    print("Error:", response.status_code, response.text)