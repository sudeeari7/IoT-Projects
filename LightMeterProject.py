# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:02:31 2019

@author: smahadeva
"""
# Logistic Regression

# Importing the libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib

state = int(input("Enter 1 to train or 2 predict : "))

if ( state == 1):
	dataset = pd.read_csv("GuidedLightMeter.csv")
	dataset.info()
    
	X = dataset.iloc[:,[0,1]].values
	Y = dataset.iloc[:,-1].values
    
    # Splitting the dataset
	x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, random_state=0)
    
	sc_x = StandardScaler()
	x_train_scaled = sc_x.fit_transform(x_train)
	x_test_scaled = sc_x.transform(x_test)
    
    # Random Forest Classifier
	classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
	classifier.fit(x_train_scaled, y_train)
	
	# Fit Logistic Regression
	#classifier = LogisticRegression()
	#classifier.fit(x_train_scaled, y_train)
    
	y_train_pred = classifier.predict(x_train_scaled)
	y_test_pred = classifier.predict(x_test_scaled)
 
    # View the results in Confusion Matrix
	cm_train = pd.DataFrame(data=confusion_matrix(y_train, y_train_pred),index=["Actual 0", "Actual 1"], columns=["Predicted 0","Predicted 1"])
	cm_test = pd.DataFrame(data=confusion_matrix(y_test, y_test_pred),index=["Actual 0", "Actual 1"], columns=["Predicted 0","Predicted 1"])
	print(cm_train)
	print(cm_test)

	joblib.dump(sc_x, 'scaler.pkl')
	joblib.dump(classifier,'classifier.pkl')


if(state == 2):
	from boltiot import Bolt
	import json
	import csv
	api_key = "XXXXXXXXXXXXXXXXXX"
	device_id = "XXXXXXXXXXX"

	mybolt = Bolt(api_key, device_id)

    # Final DF
	ptable = pd.read_csv("LightMeter.csv")
	print(" *****Aperture of F-stop varies based on the kind of photo you are clicking****")
	print("****Enter what kind of photo you want to click****")
	print("******** For Landscape enter 1 ********")
	print("******* For Group Photo enter 2 *******")
	print("********* For Potrait enter 3 *********")
	kind = int(input("Enter the kind of photo you clicking : "))
	response = mybolt.analogRead('A0')
	data = json.loads(response)
	lux = float(data['value'])
	ev = np.round(math.log(lux,2) - 3,0)
	print("Exposure Value = ",ev)
	print("luminence = ",lux)
	predict_dataset = ptable[ptable['EV']==ev]
	output_dataset = predict_dataset.copy()
	x_predict = predict_dataset.iloc[:,[0,1]].values

	sc_x = joblib.load('scaler.pkl')
	classifier = joblib.load('classifier.pkl')


	x_predict_scaled = sc_x.transform(x_predict)
	y_predict = classifier.predict_proba(x_predict_scaled)
	output_dataset['Match'] = y_predict[:,1]
	glow = int(np.round(255 - lux/4.02,0))
	value = str(glow)
	exp_comp = np.round((0-ev)/10,1)
	fc = np.round(glow/42 - 3,1)
	output_dataset['Exposure Compensation'] = exp_comp
	output_dataset['Flash'] = fc
	
	print("\nPossible settings are")
#	lower_quartile, top_quartile = output_dataset['Fstop'].quantile([.25, .50])
#	if (kind == 1):
#		possible = output_dataset[output_dataset['Fstop'] > top_quartile]
#	elif (kind == 2):
#		possible = output_dataset[(output_dataset['Fstop'] > lower_quartile) & (output_dataset['Fstop'] <= top_quartile)]
#	else:
#		possible = output_dataset[output_dataset['Fstop'] <= lower_quartile]
		
	print(output_dataset[['Fstop','ShutterSpeed','Exposure Compensation','Flash','Match']])
	
	print("\nRecomended settings is")
	final_output = output_dataset[output_dataset['Match']==output_dataset['Match'].max()]
	#final_output = final_output[['Fstop','ShutterSpeed','Exposure Compensation','Flash']]
	final_output.reset_index()
	
	
	lower_quartile, top_quartile = final_output['Fstop'].quantile([.25,.50])
	if (kind == 1):
		top_quartile = final_output['Fstop'].quartile([.50])
		recomended = final_output[output_dataset['Fstop'] >= top_quartile]
	elif (kind == 2):
		recomended = final_output[(output_dataset['Fstop'] >= lower_quartile) & (output_dataset['Fstop'] <= top_quartile)]
	else :
		recomended = final_output[output_dataset['Fstop'] <= lower_quartile]
		
	#print(value)
	print(recomended)
	mybolt.analogWrite('1',value)
	
	index = int(input("\nWhich one did you like : "))
	#app_record = pd.DataFrame([[final_output.loc[index]['Fstop'],final_output.loc[index]['ShutterSpeed'],ev,1]])
	app_record = output_dataset[['Fstop','ShutterSpeed']]
	app_record['EV'] = int(ev)
	app_record['Match'] = 0
	app_record.loc[index,'Match'] = 1
	
	with open('GuidedLightMeter.csv','a') as fd:
		writer = csv.DictWriter(fd, delimiter=',',lineterminator='\n',fieldnames=[0,1,2,3])
		for itr in list(app_record.index):
			writer.writerow({0:app_record.loc[itr]['Fstop'],1:app_record.loc[itr]['ShutterSpeed'],2:app_record.loc[itr]['EV'],3:app_record.loc[itr]['Match']})
	
	print("\n Thankyou for your feedback, we will save it for future reference\n")
	print(app_record)
