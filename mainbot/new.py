import random
import json
import pickle # used for serialization of the data as transforms in binary file and stores
import numpy as np      #it is used for mathematical and scientific computint, 
                        #it supports various array(used to alter array)#
import tensorflow as tf  # for using specific types of multidimensional array

import nltk #natural language toolkit it us used for computer to process the natural langugae
#wordnet is large english langugae database
from nltk.stem import WordNetLemmatizer # Lemmatizer is used to understand the natural word in root form.(i/p="running" rootform ="run",...)

lemmatizer = WordNetLemmatizer() #to initialize the wordenet Lemmatizer

#reading the json file we created as "intents.json"
intents = json.loads(open("intents.json").read()) 

#initializing list for words , docs and characters
words = []
classes =[]
documents = [

]
#letters that we need to ignore
ignoreLetters = ['?','!',".",","]

# now tokenizing the pattern by tokenizing the json content (nested tokenization type: jst my understanding idk what its called ;))
for intent in intents['intents']: 
    for pattern in intent["patterns"]:
        wordList = nltk.word_tokenize(pattern)  #storing each element as token by tokenization from pattern
        words.extend(wordList) #jst copy paste
        documents.append((wordList, intent['tag'])) #jst addition to words but append puts it as a single element
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters] #it will lemmatize the words and rmove the ignoring one
words = sorted(set(classes)) #its will sort the values in classes and placed in words. set will remove duplicate values

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))  # it does seriealizatioin of the data and stores in word.pkl
pickle.dump(classes, open('classes.pkl','wb'))

training = [] # to store the training data
outputEmpty = [0] * len(classes)   #This is a list of zeros, with each zero representing a class. Its length is the same as the number of classes.

for document in documents:
    bag =[] # created to generate a bag of words
    wordPatterns = document[0]    # extracts each word form teh documents 
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns] # lemmatizes the word from word patterns convers in lowercase
    for word in words: bag.append(1) if word in wordPatterns else bag.append(0) #it checks if that word exists in the wordPatterns (the lemmatized and lowercase words from the document).
                                                                                # if word it is present 1 is placed else 0 is stored

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow) #appends the concatination of the bag and o/prow

random.shuffle(training) # shuffles the training list in disordered form

training = np.array(training) # it creates an array of the training list

trainX = training[:, :len(words)]   #sliced into row 
trainY = training[:, len(words):]   #sliced into column

#sequential model is a type of model in which the program goes through several layers from input to output

model = tf.keras.Sequential() # power ful tensor flow api to create sequential model

#dense layer is the type of layer in which each lear is interconnected with each other
model.add(tf.keras.layers.Dense(128, input_shape = (len(trainX[0]),),activation = 'relu')) #it is the type of model which is used.
#now we are using dropout layer to decrease the overfitting (meaning the model will less reliable on the data(increase creativity))
model.add(tf.keras.layers.Dropout(0.5)) #it will set the inputs to 0 for the 50% input
model.add(tf.keras.layers.Dense(64 , activation = 'relu'))

model.add(tf.keras.layers.Dense(len(trainY[0]),activation ='softmax'))
