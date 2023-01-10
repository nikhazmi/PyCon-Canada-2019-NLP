#%%
# Import Libraries
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, datetime
import json

# Import Tensorflow Library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Import sklearn Library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

# Import Modules
from modules import text_cleaning, lstm_model_creation

#%%
"""
DATA DOCUMENTATION (Explanation for each number section)
    Part 1: Loading data 
        This loads the data from a URL and stores it in a Pandas DataFrame.

    Part 2: Inspect data
        This checks the structure and content of the DataFrame using the info() and head() methods. 
        It also checks for duplicates and missing values using the duplicated() and isna() methods. 
        This is useful for understanding the characteristics of the data and identifying any issues that need to be addressed.

    Part 3: Data cleaning
        This applies a custom text_cleaning function to each entry in the 'text' column of the DataFrame, and stores the cleaned text in the same column.

    Part 4: Select features and labels
        This selects the 'text' column as the feature data (X) and the 'category' column as the label data (y).

    Part 5: Data preprocessing
        This preprocesses the feature data for use in the model.

    Part 5.1: Tokenize text data
        This creates a tokenizer object that converts the text data into numerical sequences. 
        It fits the tokenizer on the X data and applies it to X to obtain the numerical sequences.

    Part 5.2: Pad sequences
        This pads the numerical sequences to have a fixed length of 300. 
        This is necessary because the input to the model must have the same shape.

    Part 5.3: One-hot encode labels
        This applies one-hot encoding to the label data. 
        ne-hot encoding converts categorical data into a numerical format that can be used as input to machine learning models.

    Part 6: Split data into train and test sets
        This splits the feature and label data into training and test sets using a 70-30 split.

    Part 7: Create and train model
        This creates a model using a custom lstm_model_creation function and trains it on the training data. 
        It also specifies callback functions for early stopping and for creating a TensorBoard log.

    Part 8: Evaluate model
        This evaluates the model's performance on the test data using classification report and a confusion matrix. 
        It also displays the confusion matrix using a ConfusionMatrixDisplay object.

    Part 9: Save models
        This saves the trained model, one-hot encoder, and tokenizer to files, allowing them to be loaded and used later without the need to retrain the model or process the data again. 
        This can be useful for deploying the model in a production environment or for continuing work on the model at a later time.
"""

# %%
#1. Data Loading
URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
df = pd.read_csv(URL)

# %%
#2. Data Inspection
df.info()
df.head()
df.duplicated().sum()
df.isna().sum()

# %%
#3. Data Cleaning
for index, text in enumerate(df['text']):
    df['text'][index] = text_cleaning(text)

# %%
#4. Features Selection
X = df['text']
y = df['category']

# %%
#5. Data Preprocessing
#5.1 Tokenizer
num_words = 5000
tokenizer = Tokenizer(num_words = num_words, oov_token = '<OOV>')

tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# %%
#5.2 Padding
X = pad_sequences(X, maxlen = 250, padding = 'post', truncating = 'post')
X = np.array(X)

# %%
#5.3 One Hot Encoder
ohe = OneHotEncoder(sparse = False)
y = ohe.fit_transform(y[::, None])

# %%
#6. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, random_state = 200, test_size = 0.25)

# %%
#7. Model Development
#7.1 Create Callback Function
es = keras.callbacks.EarlyStopping(patience = 10, verbose =0, restore_best_weights= True)
#Tensorboard callback
log_dir = 'log_dir'
tensorboard_log_dir = os.path.join(log_dir, 'overfit_demo', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = keras.callbacks.TensorBoard(tensorboard_log_dir)

# %%
#7.2 Create Model
model = lstm_model_creation(num_words, y.shape[1])
model.fit(X,y, epochs = 100, batch_size = 64, validation_data = (X_test, y_test), callbacks = [tb, es])

# %%
#8. Model Analysis
y_predicted =model.predict(X_test)

#%% 
#8.1 Make performance of the model and the reports
y_predicted = np.argmax(y_predicted, axis = 1)
y_test = np.argmax(y_test, axis = 1)

print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)

#%%
#8.2 Display the reports
disp= ConfusionMatrixDisplay(cm)
disp.plot()

# %%
#9. Save models
#9.1 Save trained tf model
model.save('Models/nlp_model.h5')

#9.2 Save encoder ohe
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

#9.3 Save tokenizer
token_json = tokenizer.to_json()
with open('token.json', 'w') as f:
    json.dump(token_json, f)
# %%
