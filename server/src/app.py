#Importing Required Libraries
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import time
import warnings
import re
import os
#from nltk.corpus import stopwords
#from nltk import word_tokenize
from tensorflow.keras.models import model_from_json
import pickle
# import flask
from flask import Flask ,request
from flask_cors import CORS, cross_origin
#STOPWORDS = set(stopwords.words('english'))
warnings.filterwarnings('ignore')

         
app = Flask(__name__)             # create an app instance
cors = CORS(app, supports_credentials=True, resources={r"/": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['POST'])                # at the end point /
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def get_response():  
    req_data = request.get_json()
    new_complaint = req_data['text']
    with open('./tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    MAX_SEQUENCE_LENGTH = 250
    # new_complaint = 'I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account do not belong to me : XXXX.'
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("model.h5")
    
    pred = loaded_model.predict(padded)
    labels = ['Credit reporting, credit repair services, or other personal consumer reports', 'Debt collection', 'Mortgage', 'Credit card or prepaid card', 'Student loan', 'Bank account or service', 'Checking or savings account', 'Consumer Loan', 'Payday loan, title loan, or personal loan', 'Vehicle loan or lease', 'Money transfer, virtual currency, or money service', 'Money transfers', 'Prepaid card']
    return({
        "label": labels[np.argmax(pred)]
        })





if __name__ == "__main__":        # on running python app.py
    app.run(debug=True,host='0.0.0.0', port=int(os.environ.get('PORT',8080)))                 # run the flask app

