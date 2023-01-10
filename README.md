## LSTM Model for NLP
This script is a natural language processing (NLP) pipeline for text classification. It starts by loading and cleaning a dataset of news articles from the BBC, which includes both the text of the articles and their corresponding categories (e.g. "sports," "politics," etc.). The script then preprocesses the text data by tokenizing it and padding the sequences to a fixed length, and one-hot encodes the categorical labels. The preprocessed data is then split into training and test sets, and a model is created and trained using a long short-term memory (LSTM) architecture. The trained model is then evaluated on the test set using a classification report and a confusion matrix, and the model and relevant preprocessing objects are saved to files.

The script also includes several helper functions and modules that are called throughout the pipeline, including a custom text_cleaning function for cleaning the text data and a lstm_model_creation function for creating the LSTM model. These functions and modules are intended to improve the readability and modularity of the code.

## Badges

![Windows 11](https://img.shields.io/badge/Windows%2011-%230079d5.svg?style=for-the-badge&logo=Windows%2011&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

## Documentation
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
        This saves the trained model, one-hot encoder, and tokenizer to files, allowing them to be loaded and used later without 
        the need to retrain the model or process the data again. 
        This can be useful for deploying the model in a production environment or for continuing work on the model at a later time.
## Report
![Display Matrix](https://user-images.githubusercontent.com/82282919/211462960-227d736b-72eb-40ed-8044-a4b1c68eaead.png)
![Report](https://user-images.githubusercontent.com/82282919/211462990-cc298d6e-152a-4bc1-8067-53e28ff15635.png)

## Graph
# Loss vs Validation
![Tensorboard](https://user-images.githubusercontent.com/82282919/211463181-d9e4a42c-91a5-4a32-8368-b4735e3c0fa8.png)

# Architecture of the Model
![model](https://user-images.githubusercontent.com/82282919/211461584-c21a3dd2-1f76-447f-beac-705321e14a29.png)

## Acknowledgement
I would like to acknowledge the following sources that were used in the development of this code:

The dataset of news articles from the BBC, which was obtained from the following URL: https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv
The text_cleaning function, which was inspired by the tutorial "Natural Language Processing with Python" by Susan Li at PyCon Canada 2019: https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial
The lstm_model_creation function, which was adapted from the tutorial "Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras" by Jason Brownlee: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
The TensorFlow and scikit-learn libraries, which were used extensively throughout the script for tasks such as tokenization, padding, one-hot encoding, model creation and training, and evaluation.
