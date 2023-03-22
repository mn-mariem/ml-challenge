import pandas as pd
import numpy as np
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle as pkl
from annoy import AnnoyIndex
import fasttext
import fasttext.util
from flask import Flask, request, render_template


# Import data
df = pd.read_csv('df_queries.csv')

# Load the embedding model
model_path = "./cc.de.300.bin"
model_embedding = fasttext.load_model(model_path)

# Load PCA file
pca = pkl.load(open("pca.pkl",'rb'))

# Load Annoy model
num_trees = 150
num_neighbors = 10
vector_dim = 100
index = AnnoyIndex(vector_dim, 'angular')
index.load('annoy_model.ann')

# For preprocessing
punctuations = string.punctuation + "«»„“‚‘‘”’´`•"
stopwords = nltk.corpus.stopwords.words('german')  


def preprocessing(text):
    no_punct = ""
    for char in text:
        if char not in punctuations :
            if char not in stopwords:
                no_punct = no_punct + char
    # Remove any remaining whitespaces
    no_punct = re.sub(r'\s+', ' ', no_punct).strip()
    # Lowercase the text
    no_punct_lower = no_punct.lower()
  
    return no_punct_lower


def get_sentence_embedding(sentence, model_embedding, vector_dim):
    words = sentence.split()
    vectors = [model_embedding.get_word_vector(word) for word in words]
    print(sentence)
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_dim)


app = Flask(__name__)

@app.route("/",methods=["POST","GET"])
def home():
    if request.method=="POST" :
        input_query = request.form["input_query"]
        input_query_preprocessed = preprocessing(input_query)
        input_vector = get_sentence_embedding(input_query_preprocessed, model_embedding, vector_dim=100)
        input_vector_pca = pca.transform(input_vector.reshape(1, -1))
        index_result_queries = index.get_nns_by_vector(input_vector_pca.reshape(-1, 1), num_neighbors)
        results_queries = [df['query'].iloc[i] for i in index_result_queries]
        
        return render_template("home.html", usr="input_query", result=True, results_queries=results_queries)
    
    return render_template("home.html")


if __name__=="__main__" :
    app.run(debug=True)
