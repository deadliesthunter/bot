import random
import json
import pickle
import numpy as np
import nltk    #natural language processing
import tensorflow


from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

import tensorflow as tf
from tensorflow.keras.models import load_model
   #uses pretend chatbot model
   

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('bot/intents.json').read())  #it will read the intents.json file
words = pickle.load(open('words.pkl','rb')) #it will open the previously created file
classes = words = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_forbeginner.h5') #pretrained chatbot is loaded using load_model

def clean_up_sentence(sentence):  #takes sentence as i/p 
    sentence_words= nltk.word_tokenize(sentence)    # sentence is tokenized into pieces
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # each piece is lemmatized
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] =1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key = lambda x:x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intents':classes[r[0]], 'probability' :str(r[1]) })
    return return_list

def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intents']  # Use 'intent' instead of 'intents'
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("Great! Bot is Running")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints,intents)
    print(res)
