import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy
import tensorflow
from tensorflow.keras.layers import Dense, Activation
import random
import pickle

import json
with open('intents.json') as file:
    data = json.load(file)

#print(data['intents'])

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        #print(wrds)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

with open("data.pickle", "wb") as f:
        pickle.dump((data, words, labels), f)

#model
model=tensorflow.keras.Sequential()
model.add(Dense(8,input_shape=(len(training[0]),)))
model.add(Dense(8))
model.add(Dense(len(output[0]), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(training, output, batch_size=10, epochs=500, shuffle=True, verbose=2)

model.save("customized_model.h5")