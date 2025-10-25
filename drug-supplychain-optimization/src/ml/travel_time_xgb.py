# src/ml/travel_time_xgb.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Dummy example: build simple dataset: distance, hour -> travel_time
rng = np.random.RandomState(0)
n=1000
distance = rng.uniform(1,30,n)
hour = rng.randint(0,24,n)
# add rush-hour effect
travel_time = distance * (1 + 0.2*((hour>=8)&(hour<=9)) + 0.3*((hour>=17)&(hour<=18))) + rng.normal(0,1,n)

df = pd.DataFrame({'distance':distance,'hour':hour,'tt':travel_time})
X = df[['distance','hour']]
y = df['tt']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
model = XGBRegressor(n_estimators=100)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test,pred))
