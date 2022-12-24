from sentence_transformers import SentenceTransformer
import re
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Input,
    Dropout,
    Activation,
    Conv1D,
    BatchNormalization,
)
from tensorflow.keras.layers import Bidirectional


def build_model():
    inputs = Input(name='the_input', shape=(1,768), dtype='float32')
    
    x = Dense(768, activation='tanh')(inputs)
    x = Dense(384, activation='tanh')(x)
    
    x = Conv1D(128, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    
    x = Dense(128, activation='tanh')(x)
    x = Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum')(x)
    x = Bidirectional(LSTM(64), merge_mode='concat')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='tanh')(x)
    
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)

def get_embedding_model(path):
    em_model = SentenceTransformer(path)
    return em_model


def clean_text(text):
    text = text.lower()
    text = re.sub("https?://\w+\.\w+\.\w+", "", text)  # removes all links
    text = re.sub("@[\w]+", "", text)  # removes usernames
    text = re.sub("#[\w]+", "", text)  # removes hashtags
    text = re.sub("rt", "", text)  # removes rt sign
    text = re.sub("[^a-z ]+", "", text)  # removes all non-alphabetic characters
    return text
