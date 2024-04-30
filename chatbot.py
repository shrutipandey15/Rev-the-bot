import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('job_intents.json', encoding='utf-8').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))


        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
# initializing training data
training = []
output_empty = [0] * len(classes)
max_words = len(words)

for doc in documents:
    bag = [0] * max_words
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in pattern_words:
        if w in words:
            bag[words.index(w)] = 1

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Append bag and output_row as tuples to ensure consistent shape
    training.append((bag, output_row))

# Shuffle training data
random.shuffle(training)

# Separate bag and output_row tuples into separate lists
train_x = [entry[0] for entry in training]
train_y = [entry[1] for entry in training]

# Convert to NumPy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

print("Training data created")

# Create and train the model
model = Sequential()
model.add(Dense(128, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

print("Model created and saved")
