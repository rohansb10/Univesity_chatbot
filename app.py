import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GlobalMaxPooling1D, Dense, Activation, Dropout, Embedding, Conv1D
import random 
import json
from keras.models import load_model

# Function to load JSON file
def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file

# Function to extract information from JSON and create DataFrame
def extract_json_info(json_file, df):
    for intent in json_file['intents']:
        for pattern in intent['patterns']:
            sentence_tag = [pattern, intent['tag']]
            df.loc[len(df.index)] = sentence_tag
    return df

# Function to preprocess input sentence
def preprocess_input_sentence(sentence, tokenizer, max_sequence_length):
    input_sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')
    return padded_sequence

# Function to get response
def get_response(sentence, loaded_model, tokenizer, max_sequence_length, intents, label_encoder):
    preprocessed_input = preprocess_input_sentence(sentence, tokenizer, max_sequence_length)
    predicted_label = loaded_model.predict(preprocessed_input).argmax(axis=-1)
    predicted_tag = label_encoder.inverse_transform(predicted_label)
    
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag[0]:
            response = random.choice(intent["responses"])
            return response
    
    return "I'm sorry, I didn't understand that."

# Load JSON file
filename = r"C:\Users\Rohan\Pictures\rohan\profile projects\chatbot\hh.json"
intents = load_json_file(filename)

# Create DataFrame
df = pd.DataFrame(columns=['Pattern', 'Tag'])
df = extract_json_info(intents, df)

# Preprocess data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Pattern'])
X = tokenizer.texts_to_sequences(df['Pattern'])
max_sequence_length = max(len(seq) for seq in X)
X_padded = pad_sequences(X, maxlen=max_sequence_length, padding='post')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Tag'])

X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Load the pre-trained model
loaded_model = load_model('chatbot.h5')

# Streamlit app
st.title("Simple Chatbot")

st.markdown("Chat with the bot by typing in the box below. Type 'exit' to end the conversation.")

# Use a unique identifier for the text input box
sentence = st.text_input("You:", key="input_text")  

if sentence.lower() == 'exit':
    st.text("Chatbot: Goodbye! Have a great day!")
else:
    response = get_response(sentence, loaded_model, tokenizer, max_sequence_length, intents, label_encoder)
    st.text("Chatbot:" + response)
