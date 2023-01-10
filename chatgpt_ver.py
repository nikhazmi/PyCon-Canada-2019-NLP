#%%
# Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os, datetime

# Import custom modules
from modules import text_cleaning, lstm_model_creation
#%%
# Load data
URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
df = pd.read_csv(URL)

# Clean text data
df['text'] = df['text'].apply(text_cleaning)

# Select features and labels
X = df['text']
y = df['category']

# Tokenize text data
num_words = 5000
tokenizer = Tokenizer(num_words = num_words, oov_token = '<OOV>')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# Pad sequences
X = pad_sequences(X, maxlen = 300, padding = 'post', truncating = 'post')
X = np.array(X)

# One-hot encode labels
ohe = OneHotEncoder(sparse = False)
y = ohe.fit_transform(y[:, None])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# Create callback functions
es = EarlyStopping(patience = 10, verbose = 0, restore_best_weights = True)
log_dir = 'log_dir'
tensorboard_log_dir = os.path.join(log_dir, 'overfit_demo', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(tensorboard_log_dir)

# Create and train model
model = lstm_model_creation(num_words, y.shape[1])
model.fit(X, y, epochs = 100, batch_size = 64, validation_data = (X_test, y_test), callbacks = [tb, es])

# Evaluate model
y_predicted = model.predict(X_test)
y_predicted = np.argmax(y_predicted, axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)

disp= ConfusionMatrixDisplay(cm)
disp.plot()
# %%
