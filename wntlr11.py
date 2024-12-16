import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 데이터 수집
ticker = 'NVDA'  # Nvidia 주식 티커
start_date = '2000-01-01'  # 엔비디아 주식의 첫 데이터부터 시작
end_date = '2024-12-16'  # 오늘 날짜

# 주식 데이터 다운로드
data = yf.download(ticker, start=start_date, end=end_date)

# 2. 데이터 전처리
data = data.dropna()  # 결측치 제거
data['Date'] = data.index
data.set_index('Date', inplace=True)

# 3. 특성 엔지니어링 (과거 60일 데이터를 사용하여 예측)
look_back = 60  # 예측할 때 과거 60일만 사용하도록 제한

# 과거의 종가를 특성으로 추가 (60일 간의 데이터만 사용)
for i in range(1, look_back + 1):
    data[f'lag_{i}'] = data['Close'].shift(i)

# 결측치 제거 (shift로 인해 첫 번째 값에 NaN이 생기므로 제거)
data = data.dropna()

# 4. 훈련 데이터 (전체 데이터 사용)
X = data[[f'lag_{i}' for i in range(1, look_back + 1)]]  # 과거 데이터를 특성으로 사용
y = data['Close']  # 종가를 예측 목표 변수로 설정

# 5. 선형 회귀 모델 훈련
model = LinearRegression()
model.fit(X, y)

# 6. 예측: 전체 데이터에 대해 예측
predictions = model.predict(X)

# 7. 예측 결과 시각화
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Actual Close Price', color='blue')  # 실제 종가
plt.plot(data.index, predictions, label='Predicted Close Price', color='orange', linestyle='--')  # 예측된 종가
plt.title('NVDA Stock Price Prediction using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
