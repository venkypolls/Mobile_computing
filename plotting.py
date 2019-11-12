import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/venkateshpolepally/education/mobile computing/CSV/car/car_1_narvekar.csv")
df1 = pd.read_csv("/Users/venkateshpolepally/education/mobile computing/CSV/car/CAR_PRACTICE_1_PATEL.csv")

wrist=df['leftElbow_x'].values
wrist1=df1['leftElbow_x'].values

#plot before
plt.figure(0)
plt.plot(wrist)
plt.plot(wrist1)
plt.show()
# print(wrist)

scaler_main=MinMaxScaler(feature_range=(-1,1))
scaler = scaler_main.fit(wrist.reshape(-1,1))
wrist=wrist.reshape(-1,1)
scaled = scaler.transform(wrist)
# print(scaled)

scaler1 = scaler_main.fit(wrist1.reshape(-1,1))
wrist1=wrist1.reshape(-1,1)
scaled1 = scaler1.transform(wrist1)

#plot after

plt.figure(1)
plt.plot(scaled)
plt.plot(scaled1)
plt.show()