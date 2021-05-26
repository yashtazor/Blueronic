import os
import pickle
import numpy as np
import pandas as pd
#import scipy.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
import seaborn as sns
from flask import Flask,render_template,request
app = Flask(__name__)

predicted_meds=[]


@app.route('/')
def hello_world():
   return render_template('index.html')

@app.route('/symptom',methods = ['POST', 'GET'])
def symptom_handle():
   if request.method == 'POST':
      result=request.form
      s1,s2,s3,s4,s5 = result['s1'],result['s2'],result['s3'],result['s4'],result['s5']
      #creating TFIDF of input symptom
      target_symptom=[[s1+' '+s2+' '+s3+' '+s4+' '+s5]]
      test_instance=pd.DataFrame(target_symptom,columns=['Mixed'])
      vectorizer = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\TFIDF.pkl','rb'))
      X=vectorizer.transform(test_instance['Mixed'])
      #testing knn on input symptoms
      neigh = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\knn.pkl','rb'))
      test_instance = pd.DataFrame.sparse.from_spmatrix(X)
      inp=test_instance.iloc[0,:]
      distances,indices = neigh.kneighbors([inp])
      #getting the predicted meds
      medlist=pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\meds.pkl','rb'))
      for i in indices.ravel():
         predicted_meds.append(medlist[i])
      filename='C:\\Users\\Ayanabha\\Blueronic\\Models\\predmed.pkl'
      pickle.dump(predicted_meds, open(filename, 'wb'))
      return render_template('categories.html',result=predicted_meds)

@app.route('/inventory',methods = ['POST', 'GET'])
def inventory_model():
   regr = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\multi_regr.pkl','rb'))
   df5 = pd.read_csv('C:\\Users\\Ayanabha\\Blueronic\\Datasets\\df_inventory.csv')
   if request.method=='POST':
      result = request.form
      med,weight = result['med'],result['weight']
      #extracting suggest count from dataset
      l1=[]
      for i in range(len(df5['Item Description'])):
         if(df5['Item Description'][i]==med):
            l1.append(df5['Suggest count'][i])
      suggest_count=round(sum(l1)/len(l1)) if len(l1)>0 else 0
      #predicting line item quantity
      x_pred=[[float(weight),float(suggest_count)]]
      y_pred=regr.predict(x_pred)
      result=[med,round(y_pred[0]),1]
      return render_template('inventory.html',result=result)
   else:
      # Making X_test for our model.
      l1 = []
      l2 = []
      l3 = []
      for i in range(len(df5['Item Description'])):
          if(df5['Item Description'][i]==predicted_meds[0] or df5['Item Description'][i]==predicted_meds[1] or df5['Item Description'][i]==predicted_meds[2]):
              l1.append(df5.iloc[i,:]['Weight (Kilograms)'])
              l2.append(df5.iloc[i,:]['Suggest count'])
              l3.append(df5.iloc[i,:]['Item Description'])   
      X_test = pd.DataFrame()
      X_test['meds']=l3
      X_test['Weight (Kilograms)'] = l1
      X_test['Suggest count'] = l2
      #predicting line item quantity using multiple regression
      y_pred = regr.predict(X_test.iloc[:,1:])
      #appending line item quantity to X_test
      X_test['Line Item Quantity']=list(y_pred)
      #finding mean line item quantity for each predicted med
      pred_med1=[]
      pred_med2=[]
      pred_med3=[]
      for i in range(len(X_test['meds'])):
         if(X_test['meds'][i]==predicted_meds[0]):
            pred_med1.append(X_test.iloc[i,:]['Line Item Quantity'])
         if(X_test['meds'][i]==predicted_meds[1]):
            pred_med2.append(X_test.iloc[i,:]['Line Item Quantity'])
         if(X_test['meds'][i]==predicted_meds[2]):
            pred_med3.append(X_test.iloc[i,:]['Line Item Quantity'])
      line_item_quantity=[round(sum(pred_med1)/len(pred_med1)) if len(pred_med1)>0 else 0,round(sum(pred_med2)/len(pred_med2)) if len(pred_med2)>0 else 0,round(sum(pred_med3)/len(pred_med3)) if len(pred_med3)>0 else 0]
      result=[predicted_meds,line_item_quantity,0]
      return render_template('inventory.html',result=result)

@app.route('/demand',methods = ['POST', 'GET'])
def demand_model():
   if request.method=='POST':
      result=request.form
      attr=result['options']
      os.system('Python C:\\Users\\Ayanabha\\Blueronic\\Scripts\\demand.py')
      message=pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\error.pkl','rb'))
      if message=='Model Success':
         result=attr
      else:
         result=message
      return render_template('demand.html',result=result)
   else:
      return render_template('demand.html')

@app.route('/production',methods = ['POST', 'GET'])
def production_model():
   model = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\random_forest.pkl','rb'))
   le = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\vendor_coder.pkl','rb'))
   if request.method=="GET":
      X=pd.read_csv('C:\\Users\\Ayanabha\\Blueronic\\Datasets\\df_demand.csv')
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
      #predicting the delivery time using random forest
      y_pred = model.predict(X_test)
      #reverse encoding the vendors
      vendors = list(le.inverse_transform(X_test['Vendor']))
      # Making our final output dataframe. (Sorted in ascending order of delivery time in days)
      df_vendor = pd.DataFrame(columns = ['Vendor', 'Days', 'Medication'])
      df_vendor['Vendor'] = vendors
      df_vendor['Days'] = list(y_pred)
      df_vendor['Medication'] = l4
      df_vendor.sort_values('Days', inplace = True)
      df_vendor.reset_index(inplace = True)
      df_vendor.drop(columns=['index'], axis = 1, inplace = True)
      sns_plot = sns.catplot(x="Medication", y="Days", hue="Vendor", kind="bar", data=df_vendor)
      sns_plot.savefig('C:\\Users\\Ayanabha\\Blueronic\\static\\production.png')
      results=[0,0]
      return render_template('production.html',result=results)
   else:
      result=request.form
      line_item_quant,freight_cost,vendor = result['quant'],result['freight'],result['vendor']
      vendor_coded=le.transform([vendor])
      test_instance=[line_item_quant,freight_cost,vendor_coded]
      df_test=pd.DataFrame([test_instance],columns=['Line Item Quantity','Freight Cost (USD)','Vendor'])
      #predicting delivery time
      y_pred=model.predict(df_test)
      results=[1,y_pred[0]]
      return render_template('production.html',result=results)

@app.route('/supply',methods = ['POST', 'GET'])
def supply_model():
   df5=pd.read_csv('C:\\Users\\Ayanabha\\Blueronic\\Datasets\\df_supply.csv')
   # Making an input dataframe X according to predicted meds.
   l1 = []
   l2 = []
   l3 = []
   l4 = []
   l5 = []

   for i in range(len(df5['Item Description'])):
       if(df5['Item Description'][i]==predicted_meds[0] or df5['Item Description'][i]==predicted_meds[1] or df5['Item Description'][i]==predicted_meds[2]):
           l1.append(df5.iloc[i,:]['Shipment Mode'])
           l2.append(df5.iloc[i,:]['Vendor'])
           l3.append(df5.iloc[i,:]['Freight Cost (USD)'])
           l4.append(df5.iloc[i,:]['Shipment Cost'])
           l5.append(df5.iloc[i,:]['db_cluster'])
           
   X_test = pd.DataFrame()
   X_test['Shipment Mode'] = l1
   X_test['Vendor'] = l2
   X_test['Freight Cost (USD)'] = l3
   X_test['Shipment Cost'] = l4
   X_test['db_cluster'] = l5
   # Make the predictions for supply score.
   regr = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\neural_regr.pkl','rb'))
   neural_pred = []
   for i in range(X_test.shape[0]):
       neural_res = regr.predict([X_test.iloc[i,:]])
       neural_pred.append(neural_res.tolist()[0])
   # Inverse transform and print the best supplier.
   le1 = pickle.load(open('C:\\Users\\Ayanabha\\Blueronic\\Models\\vendor_label.pkl','rb'))
   best_vendor_index = neural_pred.index(min(neural_pred))
   best_vendor=le1.inverse_transform(X_test['Vendor'].astype(int)).tolist()[best_vendor_index]
   supply_score=min(neural_pred)
   result=[best_vendor,supply_score]
   return render_template('supply.html',result=result)

if __name__ == '__main__':
   app.run()
