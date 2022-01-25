from sklearn.base import BaseEstimator, TransformerMixin
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from gensim.models import LdaModel
import re
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='russian', norm=None):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.norm = norm
        if self.norm is not None:
            self.lemmatizer = pymorphy2.MorphAnalyzer()
    
    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r' ',text)

    def remove_html(self, text):
        html=re.compile(r'<.*?>')
        return html.sub(r' ',text)

    def remove_mail(self, text):
        mail=re.compile(r'^([a-z0-9_\.-]+)@([a-z0-9_\.-]+)\.([a-z\.]{2,6})$')
        return mail.sub(r' ',text)

    def mytokenize(self, text):
        text = self.remove_html(text)
        text = self.remove_URL(text)
        text = self.remove_mail(text)
        text = text.lower()
        text = re.sub("[^а-яёйa-z0-9]", " ", text)
        text = re.sub("\s+", " ", text)
        text = word_tokenize(text)
        text = [word for word in text if not word in self.stopwords]
        if self.norm is not None:
            text = [self.lemmatizer.normal_forms(word)[0] for word in text]
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        return [self.mytokenize(document) for document in documents]


class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.id2word = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.id2word = Dictionary.load(self.path)

    def save(self):
        self.id2word.save(self.path)

    def fit(self, documents, labels=None):
        self.id2word = Dictionary(documents)
        self.save()
        return self

    def transform(self, documents):
        docvec = [self.id2word.doc2bow(document) for document in documents]
        return docvec

class GensimLda(BaseEstimator, TransformerMixin):

    def __init__(self, mydict, num_topics, path=None):
        self.path = path
        self.mydict = mydict
        self.num_topics = num_topics
        self.model = None

    def load(self):
        if os.path.exists(self.path):
            self.model = LsiModel.load(self.path)

    def save(self):
        self.model.save(self.path)
        
    def make_vec(self, row_matrix, num_top):
        matrix = np.zeros((len(row_matrix), num_top))
        for i, row in enumerate(row_matrix):
            matrix[i, list(map(lambda tup: tup[0], row))] = list(map(lambda tup: tup[1], row))
        return matrix

    def fit(self, documents, labels=None):
        self.model = LdaModel(documents, id2word=self.mydict, num_topics=self.num_topics)
        return self

    def transform(self, documents):
        corpus = self.model[documents]
        documents = self.make_vec(corpus, self.model.num_topics)
        return documents
