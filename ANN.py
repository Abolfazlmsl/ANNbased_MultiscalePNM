import numpy as np
import pandas as pd
from PIL import Image
import porespy as pp
import openpnm as op
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

load_files = False
resolution = 3.9 * 1e-6

output_path = "./ML_output/"
fileResultname = "FinalResult"

path = './path_to_HR_images_folder/'
image_format = '*.tif'

if load_files:
    inputData = np.load(output_path + 'inputData_' + fileResultname + '.npy', allow_pickle=True)
    outputData = np.load(output_path + 'outputData_' + fileResultname + '.npy', allow_pickle=True)
else:
    # Reading image paths
    imgs_path = glob.glob(path + image_format)
    imgs_path.sort()
    
    # Loading images
    I = Image.open(imgs_path[0])
    I = I.convert("L")
    width, height = np.shape(I)
    depth = len(imgs_path)
    
    mainImage = np.zeros(shape=[width, height, depth])
    for count, img_path in enumerate(imgs_path):
        I = Image.open(img_path)
        I = I.convert("L")
        mainImage[:, :, count] = I
    
    mainImage = np.array(mainImage, dtype=bool)
    
    totalpn = pp.networks.snow2(mainImage, voxel_size=resolution)
    totalNetwork = op.network.GenericNetwork()
    totalNetwork.update(totalpn.network)
    
    minThroatLength = np.min(totalNetwork['throat.total_length'])
    maxThroatLength = np.max(totalNetwork['throat.total_length'])
    
    with open(output_path + fileResultname, 'w') as file1:
        file1.write(f"minThroatLength = {minThroatLength} \n")
        file1.write(f"maxThroatLength = {maxThroatLength} \n")

    def find_points_between(points, point1, point2):
        min_x, max_x = min(point1[0], point2[0]), max(point1[0], point2[0])
        min_y, max_y = min(point1[1], point2[1]), max(point1[1], point2[1])
        min_z, max_z = min(point1[2], point2[2]), max(point1[2], point2[2])
        
        points_between = [p for p in points if min_x <= p[0] <= max_x and min_y <= p[1] <= max_y and min_z <= p[2] <= max_z]
        
        return points_between
    
    inputData = []
    outputData = []
    
    Ps = len(totalNetwork['pore.all'])
    
    pore_coords = totalNetwork['pore.coords']
    
    for i in range(Ps):
        for j in range(Ps):
            if j <= i:
                continue
            
            if (np.linalg.norm(totalNetwork['pore.coords'][i]-totalNetwork['pore.coords'][j]) > maxThroatLength):
                continue
    
            haveThroat = False
            throat_length = 0
            throat_diam = 0
    
            throat_mask = np.isin(totalNetwork['throat.conns'], [i, j]).all(axis=1)
            if True in throat_mask:
                throatindex = np.where(throat_mask)
                haveThroat = True
                throat_length = totalNetwork['throat.total_length'][throatindex][0]
                throat_diam = totalNetwork['throat.equivalent_diameter'][throatindex][0]
    
            pore1coord = totalNetwork['pore.coords'][i]
            pore2coord = totalNetwork['pore.coords'][j]
            pore1diam = totalNetwork['pore.equivalent_diameter'][i]
            pore2diam = totalNetwork['pore.equivalent_diameter'][j]
                 
            inputData.append([pore1coord[0], pore1coord[1],
                              pore1coord[2], pore2coord[0],
                              pore2coord[1], pore2coord[2],
                              pore1diam, pore2diam])
    
            outputData.append([haveThroat, throat_length if haveThroat else 0, throat_diam if haveThroat else 0])
        
    inputData = np.array(inputData)
    outputData = np.array(outputData)
    
    np.save(output_path + 'inputData_' + fileResultname + '.npy', inputData) 
    np.save(output_path + 'outputData_' + fileResultname + '.npy', outputData)  

# Separate labels (classification) and the continuous outputs (regression)
labels = outputData[:, 0]  
lengths = outputData[:, 1]  
diameters = outputData[:, 2]  

def balance_data(X, y, lengths, diameters):
    df = pd.DataFrame(X)
    df['label'] = y
    df['length'] = lengths
    df['diameter'] = diameters

    df_1 = df[df['label'] == 1]
    df_0 = df[df['label'] == 0]

    num_samples_1 = len(df_1)
    df_0_balanced = df_0.sample(n=num_samples_1, random_state=42)

    df_balanced = pd.concat([df_1, df_0_balanced])

    X_balanced = df_balanced.drop(['label', 'length', 'diameter'], axis=1).values
    y_balanced = df_balanced['label'].values
    lengths_balanced = df_balanced['length'].values
    diameters_balanced = df_balanced['diameter'].values
    
    return X_balanced, y_balanced, lengths_balanced, diameters_balanced

X_balanced, y_balanced, lengths_balanced, diameters_balanced = balance_data(inputData, labels, lengths, diameters)

#--------------------------------------------------------------
# Split data into Train, Validation, and Test
#--------------------------------------------------------------
# First, split into Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp, lengths_train, lengths_temp, diameters_train, diameters_temp = train_test_split(
    X_balanced, y_balanced, lengths_balanced, diameters_balanced, test_size=0.3, random_state=42)

# Split Temp into Validation (15%) and Test (15%)
X_val, X_test, y_val, y_test, lengths_val, lengths_test, diameters_val, diameters_test = train_test_split(
    X_temp, y_temp, lengths_temp, diameters_temp, test_size=0.5, random_state=42)

#--------------------------------------------------------------
# Define the unified multi-output model
#--------------------------------------------------------------

input_layer = Input(shape=(X_train.shape[1],))

shared_layer_1 = Dense(128, activation='relu')(input_layer)
shared_layer_2 = Dense(64, activation='relu')(shared_layer_1)

classification_output = Dense(1, activation='sigmoid', name='classification_output')(shared_layer_2)

length_output = Dense(1, name='length_output')(shared_layer_2)

diameter_output = Dense(1, name='diameter_output')(shared_layer_2)

multi_output_model = Model(inputs=input_layer, outputs=[classification_output, length_output, diameter_output])

multi_output_model.compile(
    loss={
        'classification_output': 'binary_crossentropy',
        'length_output': 'mean_squared_error',
        'diameter_output': 'mean_squared_error',
    },
    optimizer=Adam(),
    metrics={'classification_output': 'accuracy'}
)

# Prepare labels and outputs for training
outputs_train = {
    'classification_output': y_train,
    'length_output': lengths_train,
    'diameter_output': diameters_train
}

outputs_val = {
    'classification_output': y_val,
    'length_output': lengths_val,
    'diameter_output': diameters_val
}

multi_output_model.fit(
    X_train,
    outputs_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, outputs_val)
)

#--------------------------------------------------------------
# Evaluate and predict using the model
#--------------------------------------------------------------
outputs_test = {
    'classification_output': y_test,
    'length_output': lengths_test,
    'diameter_output': diameters_test
}

evaluation = multi_output_model.evaluate(X_test, outputs_test)
print(f"Evaluation Results: {evaluation}")

predictions = multi_output_model.predict(X_test)

classification_preds = (predictions[0] >= 0.5).astype(int)
length_preds = predictions[1] 
diameter_preds = predictions[2]  

# Metrics
classification_accuracy = accuracy_score(y_test, classification_preds)
length_mse = mean_squared_error(lengths_test, length_preds)
diameter_mse = mean_squared_error(diameters_test, diameter_preds)

print(f'Classification Accuracy: {classification_accuracy}')
print(f'Mean Squared Error for Lengths: {length_mse}')
print(f'Mean Squared Error for Diameters: {diameter_mse}')

multi_output_model.save(output_path + 'trained_ann_model.h5')
