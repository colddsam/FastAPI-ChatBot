from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import random
import json

app = FastAPI()

async def load_model_async():
    return load_model('./training.h5')  

async def load_tokenizer_async():
    with open('./tokenizer.pickle', 'rb') as handle:  
        return pickle.load(handle)

async def load_num_index_async():
    with open('./num.pickle', 'rb') as handle:  
        return pickle.load(handle)

async def load_dataset_async():
    with open('./dataset.json', 'r') as outputfile:  
        return json.load(outputfile)

async def get_chatbot_response(user_input):
    print(user_input)
    MAX_SEQUENCE_LENGTH = 6

    model = await load_model_async()
    tokenizer = await load_tokenizer_async()
    num_index = await load_num_index_async()
    dataset = await load_dataset_async()

    input_sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    predictions = model.predict(padded_sequence)
    predicted_intent_index = np.argmax(predictions)
    idx = index_mapping(predicted_intent_index, num_index)
    
    return json_reply(idx, dataset)

def index_mapping(idx, num_index):
    for key, val in num_index.items():
        if val == idx:
            return key
    return 'none'

def json_reply(idx, dataset):
    replies = []
    for intent in dataset['intents']:
        if intent['tag'] == idx:
            replies.extend(intent['responses'])
    return random.choice(replies)

class scoringItem(BaseModel):
    prompt: str
    time: str

@app.post('/')
async def scoring_endpoint(item: scoringItem):
    prompt = item.prompt
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    
    try:
        reply = await get_chatbot_response(prompt)
        return {"question": prompt, "reply": reply, "time": curr_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
