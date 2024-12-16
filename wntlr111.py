import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

# 3. 종가(Close) 데이터만 사용
data_close = data['Close']

# 4. 데이터 정규화 (LSTM은 정규화된 데이터를 더 잘 학습함)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close.values.reshape(-1, 1))

# 5. 과거 60일 데이터를 입력으로 사용하여 예측
look_back = 60

# X와 y 데이터를 생성 (과거 60일 데이터를 사용)
X = []
y = []

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])  # 과거 60일 데이터
    y.append(scaled_data[i, 0])  # 예측하고자 하는 값 (현재일 종가)

X, y = np.array(X), np.array(y)

# LSTM 모델에 맞게 데이터 모양 변경 (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 6. LSTM 모델 구축
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # 종가를 예측하기 위한 출력층

model.compile(optimizer='adam', loss='mean_squared_error')

# 7. 모델 훈련
model.fit(X, y, epochs=10, batch_size=32)

# 8. 전체 데이터를 예측 (과거 데이터와 예측된 데이터를 결합)
predicted_stock_price = model.predict(X)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)  # 역변환하여 원래 스케일로 되돌림

# 9. 1년 후 예측
# 마지막 60일의 데이터를 사용하여 1년(252일) 후 예측을 시작
last_60_days = scaled_data[-look_back:]  # 마지막 60일 데이터를 가져옴
predicted_1_year = []

for _ in range(252):  # 252 trading days (1년)
    # 모델에 입력으로 마지막 60일의 데이터를 넣고 예측
    pred_input = last_60_days.reshape(1, look_back, 1)
    pred_price = model.predict(pred_input)
    predicted_1_year.append(pred_price[0, 0])

    # 예측된 값은 마지막 60일 데이터에 추가하고, 가장 오래된 값을 제거
    last_60_days = np.append(last_60_days[1:], pred_price, axis=0)

predicted_1_year = scaler.inverse_transform(np.array(predicted_1_year).reshape(-1, 1))  # 역변환하여 원래 스케일로 되돌림

# 10. 예측 결과 시각화
future_dates = pd.date_range(start=data.index[-1], periods=252, freq='B')  # 1년 동안의 날짜 생성

plt.figure(figsize=(14, 7))
plt.plot(data.index, data_close, label='Actual Close Price', color='blue')  # 실제 종가
plt.plot(future_dates, predicted_1_year, label='Predicted Close Price for the Next Year', color='orange', linestyle='--')  # 예측된 종가
plt.title('NVDA Stock Price Prediction for the Next Year using LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
