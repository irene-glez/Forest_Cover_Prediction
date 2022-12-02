
import pandas as pd
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import lightgbm as lgb
import pickle


# Correct directory
#os.chdir

# Read train dataset
train = pd.read_csv('/Users/IRENE/Desktop/GITHUB/Forest_Cover_Prediction/src/data/train.csv')

# Feature engineering
train = train.drop(columns = ['Soil_Type7', 'Soil_Type15', 'Soil_Type25', 'Soil_Type8', 'Soil_Type28', 'Soil_Type36', 'Soil_Type9', 
'Soil_Type27', 'Soil_Type21', 'Soil_Type34', 'Soil_Type37', 'Soil_Type19','Soil_Type26', 'Soil_Type18','Soil_Type5', 'Soil_Type11', 
'Soil_Type16', 'Soil_Type20', 'Id'])

train['eDist_to_Hydrology'] = (train['Horizontal_Distance_To_Hydrology']**2 + train['Vertical_Distance_To_Hydrology']**2)**0.5 
train['human_presence'] = train['Horizontal_Distance_To_Roadways'] + train['Horizontal_Distance_To_Fire_Points']
train['Total_Hillshade_mean'] = (train['Hillshade_9am'] + train['Hillshade_3pm'] + train['Hillshade_Noon']) / 3
train['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in train['Horizontal_Distance_To_Roadways']]
train['Hillshade_Noon_Elevation_ratio'] = train['Hillshade_Noon']/(train['Elevation']+1)
train['Hillshade_9am_Elevation_ratio'] = train['Hillshade_9am']/(train['Elevation']+1)

# # Divide features and target
X = train.drop(['Cover_Type'], axis= 1)
y = train['Cover_Type']

# Train the Classifier model with the chosen params
model = lgb.LGBMClassifier(learning_rate=0.2, num_leaves=40, objective='multiclass')
transformer = QuantileTransformer()
trans_X = transformer.fit_transform(X)
model.fit(X,y)

# Save it as pickle
filename = 'final_model'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(model, archivo_salida)
