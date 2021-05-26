# # Demand Estimation

# In[ ]:
import pandas as pd
import pickle
import matplotlib.pyplot as plt

predicted_meds = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\predmed.pkl','rb'))
df5=pd.read_csv('C:\\Users\\Ayanabha\\Blueronic\\Datasets\\df5_mod.csv')

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
    plt.xlabel('Day')
    plt.ylabel('Line Item Quantity')
    plt.savefig('C:\\Users\\Ayanabha\\Blueronic\\static\\quantity.png')


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
    plt.xlabel('Day')
    plt.ylabel('Line Item Value')
    plt.savefig('C:\\Users\\Ayanabha\\Blueronic\\static\\value.png')
    print('In Model')
    pickle.dump('Model Success', open('C:\\Users\\Ayanabha\\Blueronic\\Models\\error.pkl', 'wb'))
else:
    print('In Model')
    pickle.dump('Model Error', open('C:\\Users\\Ayanabha\\Blueronic\\Models\\error.pkl', 'wb'))
