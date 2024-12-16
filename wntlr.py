import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. 데이터 수집
ticker = 'NVDA'  # Nvidia 주식 티커
start_date = '2020-01-01'
end_date = '2025-10-30'  # 오늘 날짜

# 주식 데이터 다운로드
data = yf.download(ticker, start=start_date, end=end_date)

# 2. 데이터 전처리
data = data.dropna()  # 결측치 제거
data['Date'] = data.index
data.set_index('Date', inplace=True)

# 빈도 정보 설정
data.index = pd.date_range(start=data.index[0], periods=len(data), freq='B')  # 'B'는 영업일 빈도

# 3. ARIMA 모델 피팅
model = ARIMA(data['Close'], order=(5, 1, 0))  # ARIMA 모델의 차수 설정
model_fit = model.fit()

# 4. 예측
forecast = model_fit.forecast(steps=252)  # 향후 252일(약 1년) 예측
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=252, freq='B')  # 영업일 기준

# 예측 결과를 DataFrame으로 변환
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Predicted'])

# 예측 값에 NaN이 있는지 확인
print(forecast_df.head())  # 예측된 값이 NaN이 아닌지 확인

# 5. NaN 값 제거 (예측 값에 NaN이 있는 경우)
forecast_df = forecast_df.dropna()  # NaN 값이 있다면 제거

# 6. 결과 시각화
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Historical Close Price', color='blue')
plt.plot(forecast_df, label='Predicted Close Price', color='orange')
plt.title('NVDA Stock Price Prediction for 1 Year')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()  # 그래프를 표시
