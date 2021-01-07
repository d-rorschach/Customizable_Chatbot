import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import random
import pickle

def bag_of_words(s, words):
    #bag = [0 for _ in range(len(words))]
    bag= numpy.zeros(len(words), dtype=int)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    
    return numpy.array(bag)


def chat(words, labels):
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        op=bag_of_words(inp, words)
        op=op.reshape(1,-1)
        results = model.predict(op,batch_size=1)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


with open("data.pickle", "rb") as f:
    data, words, labels = pickle.load(f)


model=tensorflow.keras.models.load_model("customized_model.h5")
chat(words, labels)