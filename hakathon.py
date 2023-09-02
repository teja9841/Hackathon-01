#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np , pandas as pd , matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"C:\Users\ASUS\Downloads\uber_rides_data.xlsx - sample_train.csv")
data.head()


# In[3]:


data.info()


# In[4]:


x = data['fare_amount'].dropna()
x.mean()


# In[5]:


data.head(2)


# ## Which of the following is the correct syntax to convert 'pickup_datetime' to datetime datatype?

# In[6]:


data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])


# ## Calculate distance between each pickup and dropoff points using Haversine formula. What is the median haversine distance between pickup and dropoff location according to the given dataset? 

# In[7]:


# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

# Calculate Haversine distances for each record
data['haversine_distance'] = data.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

# Calculate the median Haversine distance
median_distance = data['haversine_distance'].median()
print("Median Haversine Distance:", median_distance, "kilometers")


# In[8]:


data.head(2)


# ## What is the maximum haversine distance between pickup and dropoff location according to the given dataset?

# In[9]:


# Max distance:
data['haversine_distance'].max()


# ## How many rides have 0.0 haversine distance between pickup and dropoff location according to the given dataset?

# In[10]:


count_zero_distance_rides = len(data[data['haversine_distance'] == 0.0])

print(f'{count_zero_distance_rides} rides have 0.0 haversine distance between pickup and dropoff location according to the given dataset')


# ## What is the mean 'fare_amount' for rides with 0 haversine distance?

# In[11]:


mean_fare_amount_zero_distance = data[data['haversine_distance'] == 0.0]['fare_amount'].mean()

print("Mean 'fare_amount' for rides with 0.0 Haversine distance:", mean_fare_amount_zero_distance)


# + Rides with a Haversine distance of 0.0 are unusual and could indicate problems with the data
# + Zero distance suggests that the pickup and dropoff locations are the same, which could be due to incorrect data entry, missing GPS coordinates etc.

# ## What is the maximum 'fare_amount' for a ride?

# In[12]:


max_amt = data['fare_amount'].max()
print(f'Max fare amount : {max_amt}')


# ## What is the haversine distance between pickup and dropoff location for the costliest ride?

# In[13]:


# Sort the dataset by 'fare_amount' in descending order and retrieve the top row
costliest_ride = data.sort_values(by='fare_amount', ascending=False).iloc[0]

pickup_coords = (costliest_ride['pickup_latitude'], costliest_ride['pickup_longitude'])
dropoff_coords = (costliest_ride['dropoff_latitude'], costliest_ride['dropoff_longitude'])


haversine_distance = haversine(*pickup_coords, *dropoff_coords)

print("Haversine Distance for Costliest Ride:", haversine_distance, "kilometers")


# ## How many rides were recorded in the year 2014?
# 

# In[14]:


rides_2014 = data[data['pickup_datetime'].dt.year==2014]

print("Number of rides in the year 2014:", len(rides_2014))


# ## How many rides were recorded in the first quarter of 2014?

# In[15]:


rides_1st_q = data[(data['pickup_datetime'] >= '2014-1-1') & (data['pickup_datetime'] <= '2014-04-1')]

print("Number of rides in the 1st quarter of year 2014:", len(rides_1st_q))


# ## On which day of the week in September 2010, maximum rides were recorded ?

# In[16]:


data['ride_week_day'] = data['pickup_datetime'].dt.day_name()


# In[17]:


data.head(2)


# In[18]:


rides_september_2010 = data[(data['pickup_datetime'] >= '2010-09-01') & (data['pickup_datetime'] <= '2010-10-1')]


day_of_week_counts = rides_september_2010['ride_week_day'].value_counts()


max_day = day_of_week_counts.idxmax()
max_count = day_of_week_counts[max_day]

print(f"On {max_day} in September 2010, the maximum number of rides were recorded: {max_count} rides")


# # Apply a Machine Learning Algorithm to predict the fare amount given following input features:

# In[19]:


data = data.dropna()
data.isna().sum()


# In[23]:


data.head(5)


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,RobustScaler,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[43]:


# Split the data into features (X) and the target variable (y)
cont_col = ['passenger_count', 'haversine_distance']
nom_col = [ 'ride_week_day']

x = data[cont_col+nom_col]
y = data['fare_amount']

# Perform a 70-30 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Preprocessing the data

preprocessor = ColumnTransformer(transformers = [
            ('cont_pipeline' , Pipeline([
               ( 'cont_scale', StandardScaler())
            ]), cont_col),
            
            ('nom_pipeline', Pipeline([
                ('nom_encode',OneHotEncoder())
            ]),nom_col)
],remainder = 'passthrough')

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)


# In[44]:


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regression': RandomForestRegressor(),
    'KNN Regressor': KNeighborsRegressor(),
    'Decission Tree Regressor':DecisionTreeRegressor()
}

adjusted_r2_values = {}

for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = x_test.shape[1]
    
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    adjusted_r2_values[model_name] = adjusted_r2

# Find the algorithm with the least adjusted R-squared value
least_adjusted_r2_algorithm = min(adjusted_r2_values, key=adjusted_r2_values.get)

print("Adjusted R-squared values:")
for model_name, adjusted_r2 in adjusted_r2_values.items():
    print(f"{model_name}: {adjusted_r2}")

print("Algorithm with the least adjusted R-squared value:", least_adjusted_r2_algorithm)

