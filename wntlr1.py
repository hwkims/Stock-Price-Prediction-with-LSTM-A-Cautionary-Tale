import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

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

# 5. XGBoost 모델 훈련
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, verbosity=1)
model.fit(X, y)

# 6. 예측: 최신 데이터로부터 미래 예측
# 최신 데이터 (현재 날짜)에서 예측을 시작
last_known_data = data.iloc[-look_back:]  # 마지막 60일 데이터 추출

# X_latest를 과거 60일 데이터로 설정 (특성 선택만 정확히)
X_latest = last_known_data[[f'lag_{i}' for i in range(1, look_back + 1)]].values  # reshape 제거

# 예측 (예를 들어, 1일 후의 가격을 예측)
future_pred = model.predict(X_latest[-1:].reshape(1, -1))  # 마지막 1일 데이터를 예측
print(f"Predicted Price for the next day: {future_pred[0]}")

# 7. 예측 결과 시각화
# 전체 데이터에 대해 예측 결과를 시각화
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Actual Close Price', color='blue')

# 예측 결과를 추가 (오늘 이후 예측 결과)
plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Start')

# 예측 결과를 오늘 이후로 나타냄
pred_dates = pd.date_range(start=data.index[-1], periods=30, freq='B')  # 30일 간의 예측
pred_prices = [future_pred[0]]
for i in range(1, len(pred_dates)):
    # 예측된 값을 기반으로 새로운 데이터를 만듬
    new_data = last_known_data.copy()
    new_data = pd.concat([new_data, new_data.iloc[-1:]])  # append 대신 pd.concat 사용
    new_data.iloc[-1, 0] = pred_prices[-1]  # 예측된 가격을 다음 예측에 입력

    # 60일치 특성을 다시 추출
    X_latest = new_data[[f'lag_{i}' for i in range(1, look_back + 1)]].values
    future_pred = model.predict(X_latest[-1:].reshape(1, -1))  # 예측할 마지막 데이터를 reshape
    pred_prices.append(future_pred[0])

# 예측 결과 그리기
plt.plot(pred_dates, pred_prices, label='Predicted Price (Next 30 Days)', color='orange')
plt.title('NVDA Stock Price Prediction using XGBoost')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
