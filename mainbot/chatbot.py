import random
import json
import pickle
import numpy as np
import nltk    #natural language processing

from nltk.stem import WordNetLemmatizer
from keras.models import load_model   #uses pretend chatbot model

lemmatizer = WordNetLemmatizer(0)

intents = json.loads(open('bot/intents.json').read())  #it will read the intents.json file
words = pickle.load(open('words.pkl','rb')) #it will open the previously created file
classes = words = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_forbeginner.h5') #pretrained chatbot is loaded using load_model

def clean_up_sentence(sentence):  #takes sentence as i/p 
    sentence_words= nltk.word_tokenize(sentence)    # sentence is tokenized into pieces
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # each piece is lemmatized
    return sentence_words

