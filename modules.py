import re
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

def text_cleaning(text):
    """this function clean/remove text with anomlies such as 
    URLS, @NAME, NAME (Reuters) and also to conver text to lowecase.

    Args:
        text (str): Raw text

    Returns:
        text: Cleaned text
    """
    
    # have @.... remove
    text = re.sub('@[^\s]+', '', text)
    # new header remove
    text = re.sub('^.*?\)-\s*-', '', text)
    # bracket [] remove
    text = re.sub('\[.*?EST\]', '', text)
    # have URL remove
    text = re.sub('bit.ly/\d\w{1,10}', '', text)
    #converting to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    return text

def lstm_model_creation(num_words, nb_classes, embedding_layer = 128, dropout = 0.3, num_neurons = 64):
    """This function creates LSTM models with embedding layer, 2 LSTM layers, with dropout and _summary_

    Args:
        num_words (int): number of vocabulary
        nb_classes (int): number of class
        embedding_layer (int, optional): the number of output embedding llayer. Defaults to 64.
        dropout (float, optional): the rate of dropout. Defaults to 0.3.
        num_neurons (int, optional): number of rbain cells. Defaults to 64.

    Returns:
        model: returns the model created using sequential API.
    """

    model = Sequential()
    model.add(Embedding(num_words, embedding_layer))
    model.add(LSTM(embedding_layer, return_sequences= True))
    model.add(Dropout(dropout))
    model.add(LSTM(embedding_layer, return_sequences= True))
    model.add(Dropout(dropout))
    model.add(LSTM(num_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, activation = 'softmax'))
    model.summary()

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['acc'])
    plot_model(model,show_shapes=True,show_layer_names=True)
    return model