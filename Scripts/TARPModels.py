
# coding: utf-8

# # Importing Libraries

# In[1]:

# Trivial libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


# In[2]:

# Additional libraries.

#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#!pip install geopy


# In[ ]:




# # Predicting Meds

# In[3]:

# Importing the VAERS report dataset.
df1 = pd.read_csv("C://Users//Ayanabha//Blueronic//Datasets//2021VAERSData.csv", encoding='latin1')

# Removing columns with 90% NaN values.
df1.drop(["RPT_DATE", "V_FUNDBY", "CAGE_MO", "BIRTH_DEFECT", "RECVDATE", "TODAYS_DATE", "SPLTTYPE", "SYMPTOM_TEXT"], axis=1, inplace=True)

# Mean imputation on numerical columns.
df1['AGE_YRS'].fillna(round(df1['AGE_YRS'].mean()), inplace = True)
df1['CAGE_YR'].fillna(round(df1['CAGE_YR'].mean()), inplace = True)
df1['HOSPDAYS'] = df1.apply(
    lambda row: 0 if pd.isnull(row['HOSPITAL']) else row['HOSPDAYS'],
    axis = 1
)

df1['HOSPDAYS'].fillna(round(df1['HOSPDAYS'].mean()), inplace = True)
df1['NUMDAYS'].fillna(round(df1['NUMDAYS'].mean()), inplace = True)

# Filling categorical columns with mode.
df1.fillna(df1.mode().iloc[0], inplace = True)
df1


# In[3]:

# Importing the VAERS sympotoms and vaccine datasets.
df2 = pd.read_csv("C://Users//Ayanabha//Blueronic//Datasets//2021VAERSSYMPTOMS.csv", encoding='latin1')
df3 = pd.read_csv("C://Users//Ayanabha//Blueronic//Datasets//2021VAERSVAX.csv", encoding='latin1')

# Pre-processing both the datasets.
df2 = df2.dropna(axis = 0)
df2.drop(["SYMPTOMVERSION1", "SYMPTOMVERSION2", "SYMPTOMVERSION3", "SYMPTOMVERSION4", "SYMPTOMVERSION5"], axis=1, 
         inplace=True)

df3.drop(["VAX_LOT", "VAX_SITE"], axis=1, inplace=True)
df3.fillna(df3.mode().iloc[0], inplace = True)


# In[4]:

# Making the final patient dataset.
# Would be used to predict the medical needs of a particular patient.

df4 = pd.concat([df1, df2, df3], axis=1, join="inner")

# Some pre-processing.
df4.reset_index(inplace = True)
df4.drop(["DIED","VAX_TYPE","VAX_NAME","VAX_ROUTE","CAGE_YR"], axis=1, inplace=True)
df4


# In[5]:

# Importing the supply chain dataset.

df5 = pd.read_csv("C://Users//Ayanabha//Blueronic//Datasets//SCMS_Delivery_History_Dataset.csv")
df5


# In[6]:

# Make X & Y for meds prediction.

X = df4[['SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5']].copy()
Y = df4[['OTHER_MEDS']].copy()


# In[7]:

X['Mixed'] = X[X.columns[0:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X['Mixed'])

filename='C:\\Users\\Ayanabha\\Blueronic\\Models\\TFIDF.pkl'
pickle.dump(vectorizer, open(filename, 'wb'))


# In[8]:

# Converting X from Sparse Matrix to Pandas Dataframe.

import scipy.sparse
X = pd.DataFrame.sparse.from_spmatrix(X)
X


# In[9]:

# Using the elbow method to get the optimum number of clusters.

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[10]:

# Perform K-Means Clustering of the data & predict the clusters.

kmeans = KMeans(n_clusters=3, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
y_kmeans = kmeans.fit_predict(X)
y_kmeans


# In[11]:

# Counting number of elements in each cluster.

temp = dict()

for i in y_kmeans.tolist():
    temp[i] = 0
    
for i in y_kmeans.tolist():
    temp[i] += 1
    
print(temp)


# In[12]:

# Plotting the clusters.

plt.figure(figsize=(15, 10))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label = 'Centroids')
plt.show()


# In[13]:

# Appending cluster labels.

df4['Cluster'] = y_kmeans.tolist()
df4


# In[14]:

# Making a column for meds.

l=[]

for i in df4['OTHER_MEDS']:
    if(',' in i):
        temp=i.split(',')
        
        for j in range(len(temp)):
            temp[j]=temp[j].strip()
        
        l.extend(temp)
    elif(' ' not in i):
        l.append(i.strip())
        
l=list(set(l))

import random
meds=[random.choice(l) for i in range(len(df4))]
df4['meds']=meds
df4


# In[15]:

medlist=list(df4['meds'])
filename='C:\\Users\\Ayanabha\\Blueronic\\Models\\meds.pkl'
pickle.dump(medlist, open(filename, 'wb'))


# In[16]:

# Taking user medicine input.
inp = X.iloc[0, :]


# In[17]:

# KNN algorithm for suggesting meds.

from sklearn.neighbors import KNeighborsClassifier

k = 3

neigh = KNeighborsClassifier(n_neighbors = k)
neigh.fit(X, y_kmeans.tolist())

filename='C:\\Users\\Ayanabha\\Blueronic\\Models\\knn.pkl'
pickle.dump(neigh, open(filename, 'wb'))

distances, indices = neigh.kneighbors([inp])

print(indices)


# In[18]:

# Making a list for predicted meds.

predicted_meds = []

for i in indices.ravel():
    predicted_meds.append(df4['meds'][i])


# In[19]:

# Randomly assigning meds and adding them to df5.

import random

meds = [random.choice(l) for i in range(len(df5))]
df5['Item Description'] = meds


# In[20]:

# Making the weight column numeric.

x = df5['Weight (Kilograms)']

total = 0
count = 0
for i in x:
    if i.isnumeric():
        total += float(i)
        count += 1
average = total/count

for i in range(len(x)):
    if not x[i].isnumeric():
        x[i] = average
        


# In[21]:

df5['Weight (Kilograms)'] = x


# In[22]:

x2 = set(df4['meds'])
x2


# In[23]:

# Appending meds suggestion count to a file.

a = open('C://Users//Ayanabha//Blueronic//Medications//Medications.txt', 'a')

for i in x2:
    a.write(str(i+'$' + str(random.randint(1, 1000)) + '\n'))

a.close()


# In[24]:

# Storing data from file in a dictionary.

l = dict()
a = open('C://Users//Ayanabha//Blueronic//Medications//Medications.txt', 'r')

Lines = a.readlines()
 
# Strips the newline character
for line in range(len(Lines)-1):
    temp = Lines[line].split('$')
    l[temp[0]] = int(temp[1].replace('\n','').strip())

a.close()
    


# In[25]:

# Making an item description column in df5.

x1 = []

for i in df5['Item Description']:
    if i in l.keys():
        x1.append(l[i])
    else:
        x1.append(np.nan)

df5['Suggest count'] = x1
df5['Suggest count'].fillna(df5['Suggest count'].median(), inplace=True)
df5


# In[ ]:




# # Inventory Management Model

# In[26]:

# TODOs.

# 1. Subset the rows corresponding to predicted meds to be used as data for MLR. 


# In[27]:

# Multiple linear regression.

from sklearn import linear_model

X = df5[['Weight (Kilograms)', 'Suggest count']].astype(float)
y = df5['Line Item Quantity']

regr = linear_model.LinearRegression()
regr.fit(X, y)


# In[28]:

x_pred = [[float(df5['Weight (Kilograms)'][0]), float(df5['Suggest count'][0])]]
regr.predict(x_pred)


# In[29]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[30]:

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


# In[31]:

y_pred = regr.predict(X_test)


# In[32]:

print(y_pred)


# In[33]:

from sklearn.metrics import r2_score, mean_absolute_error
r2_score(y_test, y_pred)

mean_absolute_error(y_test, y_pred)


# In[ ]:




# # Demand Estimation

# In[ ]:

predicted_meds = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\predmed.pkl','rb'))


# In[34]:

# Making a new dataframe (df6) of predicted meds instances.

df6=pd.DataFrame(columns=['Scheduled Delivery Date','Line Item Quantity','Line Item Value'])

l1=[]
l2=[]
l3=[]

for i in range(len(df5['Item Description'])):
    if(df5['Item Description'][i]== predicted_meds[0] or df5['Item Description'][i]==predicted_meds[1] or df5['Item Description'][i]==predicted_meds[2]):
        print(df5.iloc[i,:]["Line Item Quantity"])
        l1.append(df5.iloc[i,:]['Scheduled Delivery Date'])
        l2.append(df5.iloc[i,:]['Line Item Quantity'])
        l3.append(df5.iloc[i,:]['Line Item Value'])

df6['Scheduled Delivery Date']=l1
df6['Line Item Quantity']=l2
df6['Line Item Value']=l3        

df6


# In[35]:

# Preprocessing the dataset (df6) for 'Demand Estimation'.

df6["Scheduled Delivery Date"]= pd.to_datetime(df6["Scheduled Delivery Date"],format='%d-%b-%y')
df6 = df6.sort_values(by="Scheduled Delivery Date")
df6.reset_index(inplace=True)
df6.drop(['index'],axis=1,inplace=True)
df6


# In[36]:

# ARIMA forecasting of line item quantity and line item value.

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

if(df6.shape[0]>=3):
 
    casemodel = ARIMA(df6['Line Item Quantity'], order=(1, 0, 0))
    casemodel_fit = casemodel.fit(disp=0)


    # In[37]:

    # Forecasting line item quantity using ARIMA
    Y_forecast_arima = casemodel_fit.forecast(steps = df6.shape[0])[0]
    Y_forecast_arima


    # In[38]:

    # Plotting the original and forecasted line item quantity.

    plt.plot(df6.index,df6['Line Item Quantity'])
    plt.plot(df6.index,list(Y_forecast_arima),color='red')
    plt.show()


    # In[39]:

    # ARIMA forecasting of line item value.

    casemodel = ARIMA(df6['Line Item Value'], order=(1, 0, 0))
    casemodel_fit = casemodel.fit(disp=0)
    Y_forecast_arima = casemodel_fit.forecast(steps = df6.shape[0])[0]
    Y_forecast_arima


    # In[40]:

    # Plotting the original and forecasted line item value.

    plt.plot(df6.index,df6['Line Item Value'])
    plt.plot(df6.index,list(Y_forecast_arima),color='red')
    plt.show()
else:
    print('Model error')

# In[ ]:




# # Production

# In[41]:

# Making a copy of df5.

df5_copy = df5.copy(deep=True)


# In[42]:

# Preprocessing dataframe (df5) for production model.

df5["Scheduled Delivery Date"]= pd.to_datetime(df5["Scheduled Delivery Date"],format='%d-%b-%y')
d_min=min(df5['Scheduled Delivery Date'].tolist())
X=str(d_min)
df5['PO Sent to Vendor Date'].replace(to_replace=["Date Not Captured", "N/A - From RDC"], value =str(X[0:10]),inplace=True)
k = df5["Scheduled Delivery Date"] - pd.to_datetime(df5['PO Sent to Vendor Date'])
df5['Delivery Time'] = k.dt.days
df5


# In[43]:

# Making X and y from required inputs and labels.

X = df5[['Line Item Quantity', 'Freight Cost (USD)', 'Vendor']]
y = df5[['Delivery Time']]


# In[44]:

X


# In[45]:

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['Vendor'] = le.fit_transform(X['Vendor'])

X


# In[46]:

X['new'] = pd.to_numeric(X['Freight Cost (USD)'].astype(str).str.replace(',',''), errors='coerce').fillna(np.nan).astype(float)
X['new'].fillna(X['new'].mean(), inplace = True)
X['Freight Cost (USD)'] = X['new']
X.drop(columns=['new'], inplace = True)
X['meds'] = df5['Item Description']
X


# In[47]:

# Random Forest Regression for predicting delivery time.

from sklearn.ensemble import RandomForestClassifier

num_trees = 5
max_features = 3

model = RandomForestClassifier(n_estimators = num_trees,max_features=max_features)
model.fit(X.iloc[:, :3], y)


# In[48]:

# Making X_test for our model.

l1 = []
l2 = []
l3 = []
l4 = []

for i in range(len(X['meds'])):
    if(X['meds'][i]==predicted_meds[0] or X['meds'][i]==predicted_meds[1] or X['meds'][i]==predicted_meds[2]):
        l1.append(X.iloc[i,:]['Line Item Quantity'])
        l2.append(X.iloc[i,:]['Freight Cost (USD)'])
        l3.append(X.iloc[i,:]['Vendor'])
        l4.append(X.iloc[i,:]['meds'])
        
X_test = pd.DataFrame()

X_test['Line Item Quantity'] = l1
X_test['Freight Cost (USD)'] = l2
X_test['Vendor'] = l3

X_test


# In[49]:

# Predicting the labels for vendors with least delivery time.

y_pred = model.predict(X_test)
y_pred


# In[50]:

# Inverse transformaing our labels to actual vendors.

vendors = list(le.inverse_transform(X_test['Vendor']))
vendors


# In[51]:

# Making our final output dataframe. (Sorted in ascending order of delivery time in days)

df_vendor = pd.DataFrame(columns = ['Vendor', 'Days', 'Medication'])

df_vendor['Vendor'] = vendors
df_vendor['Days'] = list(y_pred)
df_vendor['Medication'] = l4

df_vendor.sort_values('Days', inplace = True)
df_vendor.reset_index(inplace = True)
df_vendor.drop(columns=['index'], axis = 1, inplace = True)

df_vendor


# In[52]:

# Visualizing the output.

print('\n\nThe scatter plots for the clusters across various columns are')

import seaborn as sns
from matplotlib import rcParams

a4_dims = (25, 20)
fig, ax = plt.subplots(figsize=a4_dims)
sns.catplot(ax=ax, x="Medication", y="Days", hue="Vendor", kind="bar", data=df_vendor)


# In[ ]:




# # Supply Management

# In[53]:

# Cleaning Freight Cost (USD) column in df5.

df5['Freight Cost (USD)'] = pd.to_numeric(df5['Freight Cost (USD)'].astype(str).str.replace(',',''), errors='coerce').fillna(np.nan).astype(float)
df5['Freight Cost (USD)'].fillna(df5['Freight Cost (USD)'].mean(), inplace = True)
df5


# In[54]:

# Label encoding Vendor and Shipment Mode.

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()

df5['Vendor'] = le1.fit_transform(df5['Vendor'])
df5['Shipment Mode'] = le2.fit_transform(df5['Shipment Mode'].astype(str))
df5


# In[55]:

# Calculating and adding Shipment Cost column to df5.

df5['Shipment Cost'] = df5['Freight Cost (USD)']/(df5['Weight (Kilograms)'].astype(float) * df5['Line Item Quantity'])
df5['Shipment Cost'].replace(to_replace=[np.inf], value =np.nan,inplace=True)
df5['Shipment Cost'].fillna(df5['Shipment Cost'].mean(),inplace=True)
df5


# In[56]:

# Getting the latitudes and longitudes of all manufacturing sites in df5.

geocode=dict()
sites=list(set(df5['Manufacturing Site']))

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="Blueronic")
for i in sites:
    location = geolocator.geocode(i,timeout=None)
    if location == None:
        geocode[i]=[np.nan,np.nan]
    else:
        geocode[i]=[location.latitude,location.longitude]
        
geocode


# In[57]:

# Appending latitude and longitude of vendors to df5.

latitude=[]
longitude=[]

for i in df5['Manufacturing Site']:
    latitude.append(geocode[i][0])
    longitude.append(geocode[i][1])
    
df5['lat']=latitude
df5['long']=longitude

df5['lat'].fillna(df5['lat'].mean(), inplace = True)
df5['long'].fillna(df5['long'].mean(), inplace = True)

df5


# In[58]:

# DBSCAN for spatial data (manufacturing sites).

from sklearn.cluster import DBSCAN

X = df5[['lat','long']]
db_clustering=DBSCAN().fit(X)
set(list(db_clustering.labels_))


# In[59]:

# Plotting the spatial data (manufacturing sites).

plt.figure(figsize=(25, 20))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=db_clustering.labels_, cmap='rainbow',s=200)
plt.show()


# In[60]:

# Adding the cluster assignments as a column to df5.

df5['db_cluster']=list(db_clustering.labels_)
df5


# In[61]:

# Appending a supply score column to df5.

df5['Supply score'] = df5['Shipment Mode'] + df5['Vendor'] + df5['Freight Cost (USD)'] + df5['Shipment Cost'] + df5['db_cluster']
df5


# In[62]:

# Train-test split for neural network.

X = df5[['Shipment Mode','Vendor','Freight Cost (USD)','Shipment Cost','db_cluster']]
y = df5[['Supply score']]


# In[63]:

# Regressive Neural Network for supply score.

from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)


# In[64]:

# Making an input dataframe X according to predicted meds.

l1 = []
l2 = []
l3 = []
l4 = []
l5 = []

for i in range(len(df5['Item Description'])):
    if(df5['Item Description'][i]==predicted_meds[0] or df5['Item Description'][i]==predicted_meds[1] or df5['Item Description'][i]==predicted_meds[2]):
        l1.append(X.iloc[i,:]['Shipment Mode'])
        l2.append(X.iloc[i,:]['Vendor'])
        l3.append(X.iloc[i,:]['Freight Cost (USD)'])
        l4.append(X.iloc[i,:]['Shipment Cost'])
        l5.append(X.iloc[i,:]['db_cluster'])
        
X_test = pd.DataFrame()
X_test['Shipment Mode'] = l1
X_test['Vendor'] = l2
X_test['Freight Cost (USD)'] = l3
X_test['Shipment Cost'] = l4
X_test['db_cluster'] = l5
X_test


# In[65]:

# Make the predictions for supply score.

neural_pred = []
for i in range(X_test.shape[0]):
    neural_res = regr.predict([X_test.iloc[i,:]])
    neural_pred.append(neural_res.tolist()[0])
    
neural_pred


# In[66]:

# Inverse transform and print the best supplier.

best_vendor_index = neural_pred.index(min(neural_pred))
le1.inverse_transform(X_test['Vendor'].astype(int)).tolist()[best_vendor_index]


# In[ ]:



