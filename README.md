# Blueronic

Blueronic is a Full-Stack application for optimizing the Supply Chain aspects of Healthcare Products. It works by optimizing the Supply Chain aspects under four main categories 

* Inventory Management
* Demand Estimation
* Production
* Supply Management


## Architectural Diagram and Modules
![architecture](https://user-images.githubusercontent.com/42903837/130324130-07cd3060-9ca0-4125-ba63-d7ea80739e35.PNG)


### Machine Learning Module

#### 1. Medication Prediction Model

* The medications required by the patient would depend on the symptoms exhibited by the same. The first step would be to generate a ‘Syptom2Vec’ representation of the text by passing it through a TFIDF pipeline.
* Since there is a large number of meds in our dataset corresponding to the various symptoms, classifying the correct meds for a given tuple of symptoms is bound to return results with minimal accuracy.
* A better approach would be to segregate the meds into clusters based on the similarity of the TF-IDF vectors in order to generalize the medications according to intra-cluster similarity, which helps to achieve an appreciable amount of accuracy.
* Once the cluster labels have been assigned, a KNN classifier can be trained on the vectors and labels, corresponding to which a test symptom tuple when given as input, the classifier returns the k-nearest medications relevant to that instance, which serve as the predicted meds for that instance.

#### 2. Inventory Management Model

* The core idea of this model is to predict how much of a particular medication is actually needed to be stored in the hospital inventory.
* To predict this quantity, a multiple linear regression model is built which gives this quantity as a function of the ‘weight’ of the medications and a variable which stores the number of times a particular medication is predicted.
* This ‘suggestion count’ variable prevents the cold start problem in our regressor by assigning random seed values to each medication and updating it accordingly to the medications that get predicted.
* Once the regressor is trained, the meds which are predicted from the symptoms are used to subset test instances from our dataset to be fed into the same and obtaining the optimal number of medications that need to be stored in the inventory.

#### 3. Demand Estimation Model

*	The aim is to obtain the item quantity and item value to be forecasted based on an increasing time series for the medications predicted from the symptoms.
*	ARIMA (Auto Regressive Integrated Moving Averages) model is used for this purpose. The item quantity and value is fed as input to the model and it trains on the previous instances of this dataset at each iteration to give a forecast of the future which indicates the demand of the medications that will be required.
*	This can be used to stock on the existing inventory in accordance to the predicted values.

#### 4. Production Model

*	This model predicts the delivery time required to send the medications to the hospital, calculated as a difference between the scheduled delivery date and the date on which the vendor obtained the medications, present in the dataset
*	The delivery time depends on three factors – the medicine quantity, freight cost and the vendor involved
*	Since the vendor is a categorical data, it is label encoded for model training.
*	The independent and dependent variables are fed to a random forest classifier in order to improve the accuracy of the delivery time so obtained.
*	The predicted meds are then used to subset test instances for this ensemble classifier to calculate the estimated delivery time.
*	Reverse label encoding on the vendor is done and then the medications are plotted against their delivery time and vendor to see which vendor needs to be chosen for a particular medication.

#### 5. Supply Management Model

*	This is a culmination of all supply chain factors that make a vendor reliable like shipment mode, freight cost, shipment cost and vendor.
*	An extra attribute called ‘db_cluster’ is added to this model which accounts for the geospatial distribution of supply chain manufacturing sites using DBSCAN as the underlying algorithm
*	The model is designed to predict the ‘supply score’ for each vendor, which is a linear combination of all the aforementioned attributes.
*	The attributes are fed to a regressive neural network which learns the weights for the supply chain factors and returns the estimated supply chain scores.
*	The test instances acquired from the predicted meds give the supply chain scores for the vendors and the model selects the one with the minimum score as the best vendor.

### Backend Module

#### 1. Database 
 
The database would act as our central store of data not only for the machine learning part but also the full-stack part. The database would be originally storing all four of our datasets, viz., 2021VAERSDATA, 2021VAERSSYMPTOMS, 2021VAERSVAX and the Supply Chain datasets. This module would be responsible for providing all these datasets to the machine learning module to preprocess. The optimized supply-chain output from the machine learning module would then be returned to it and the output would then be stored there in some suitable form for the frontend which would be decided later on. Since the output is now available for the frontend, it can be utilized as the backend for the frontend part to which the output can easily be routed in the form of a ‘RESPONSE’ on a user ‘REQUEST’ from the front-end.

#### 2. Server 
 
This server is mainly set up to handle the routes through which the communication would take place between the frontend and the backend, i.e., this server is solely set up to coordinate with the frontend and is not related to the machine learning module in any way. It would accept the user REQUESTS and route it through the proper route to the database from where the database can take over and get it processed from the machine learning module. As soon as it has the output returned to it by the machine learning module, it can pass it to the server and the server can in turn return a RESPONSE to the user for his requested query. Hence, in a nutshell, it would coordinate the routing functions between the frontend and the backend. 

### Frontend Module

The frontend only consists of the GUI. The GUI is developed in such a way so as to provide the users with an intuitive and user-friendly interface. Users will find it very convenient to interact with our application and get answers pertaining to all of their queries namely, availability, recommendations of numerous medicines in case the patient experiences certain side-effects after consumption of vaccine. As a consequence of this, the GUI will enhance the efficiency and ease of use for the underlying logical design of the application. The visual language introduced in the design is well-tailored to the functionalities, keeping in mind the principles of user-centered design. 


## Module-Wise Technology Stack

* Machine Learning
  * Python
* Backend
  * Flask Server
  * Firebase
* Frontend
  * HTML/CSS

## How to run?

* Clone this repository and navigate to the project directory with Anaconda Prompt or Terminal.
* Create an Anaconda Virtual Environment by running **``conda env create -f env.yml``**.
* Navigate to the **Scripts** folder and run **``python TARPModels.py``**. This has to be done only once.
* Navigate to the root directory and run **``python app.py``**.
* Access the app on localhost at port 5000.

<b> <p align = "center"> Created by Ayanabha Jana, Yash Dekate, Alpanshu Kataria, and Rohan Dastidar. </p> </b>
