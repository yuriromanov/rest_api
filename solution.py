import os
import json
import time
import pickle
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from my_custom_class import TextNormalizer, GensimVectorizer, GensimLda
from flask import Flask, jsonify, request
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request

app = Flask(__name__)
clf = None

@app.route('/')
def load_model():
    global clf
    clf = pickle.load(open('pipeline.pkl', 'rb'))
    return jsonify(status='model is loaded')


@app.route('/predict', methods=['POST'])
def predict():
    global clf

    text = json.loads(request.json)['text']
    answer = clf.predict([text])
    return jsonify(answer=str(answer))

if __name__ == '__main__':
    app.run(debug=True)
