# from utils.functions import *

import pandas as pd
import os
import math

# # Correct directory
# os.chdir('/Users/laurabarreda/Documents/The_Bridge/genre_prediction/SRC/utils') 

#os.chdir

# Read train dataset
train = pd.read_csv('/Users/IRENE/Desktop/GITHUB/Forest_Cover_Prediction/src/data/train.csv')

# 
train = train.drop(columns = ['Id', 'Soil_Type7', 'Soil_Type15', 'Soil_Type25', 'Soil_Type8', 'Soil_Type28', 'Soil_Type36', 'Soil_Type9', 'Soil_Type27', 'Soil_Type21', 'Soil_Type34'])

train['eDist_to_Hydrology'] = (train['Horizontal_Distance_To_Hydrology']**2 + train['Vertical_Distance_To_Hydrology']**2)**0.5 
train['human_presence'] = train['Horizontal_Distance_To_Roadways'] + train['Horizontal_Distance_To_Fire_Points']
train['Total_Hillshade_mean'] = (train['Hillshade_9am'] + train['Hillshade_3pm'] + train['Hillshade_Noon']) / 3
train['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in train['Horizontal_Distance_To_Roadways']]
train['Hillshade_Noon_Elevation_ratio'] = train['Hillshade_Noon']/(train['Elevation']+1)
train['Hillshade_9am_Elevation_ratio'] = train['Hillshade_9am']/(train['Elevation']+1)

print(train.shape)

# # Divide features and target
# X = train_data.drop(columns = [])
# y = train_data['genre']

# # Train the Random Forest Classifier model with the chosen params
# rfc = RandomForestClassifier(max_depth=25, min_samples_leaf=1, min_samples_split=5, n_estimators=1200)
# rfc.fit(X, y)

# # Set the name of the resulting model
# date = str(datetime.today().strftime('%y%m%d%H%M%S'))
# name = filename + date   # En file va el nombre del archivo en el que quieres guardar el modelo
# path = model_path + name   # En model_path va el path que irá delante del nombre del modelo para encontrarlo

# # Save it as pickle
# with open(path, 'wb') as archivo_salida:
#     pickle.dump(rfc, archivo_salida)