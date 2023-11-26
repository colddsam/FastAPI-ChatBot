from fastapi import FastAPI
from pydantic import BaseModel
import time

from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import random
import json

app=FastAPI()

def index_mapping(idx):
    with open('num.pickle', 'rb') as handle:
        num_index = pickle.load(handle)
    for key, val in num_index.items():
        if val == idx:
            return key
    return 'none'

def json_reply(idx):
    with open('dataset.json', 'r') as outputfile:
        dataset = json.load(outputfile)
    replies = []
    for intent in dataset['intents']:
        if (intent['tag'] == idx):
            for reply in intent['responses']:
                replies.append(reply)
    return random.choice(replies)


def get_chatbot_response(user_input):
    print(user_input)
    MAX_SEQUENCE_LENGTH = 6
    model = load_model('training.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    input_sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(padded_sequence)
    predicted_intent_index = np.argmax(predictions)
    idx = index_mapping(predicted_intent_index)

    return json_reply(idx)

class scoringItem(BaseModel):
    prompt : str
    time : str

@app.post('/')
async def scoring_endpoint(item:scoringItem):
    prompt=item.prompt
    curr_time=time.strftime("%H:%M:%S" ,time.localtime())
    reply= get_chatbot_response(prompt)
    return {"question":prompt,
            "reply":reply,
            "time": curr_time}