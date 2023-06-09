import pandas as pd
import numpy as np
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import pickle as pkl
import fasttext
import fasttext.util


# ------------------------------------------------------- Preprocessing -------------------------------------------------------

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


# ------------------------------------------------------- Embedding -------------------------------------------------------

def get_sentence_embedding(sentence, model_embedding, vector_dim):
    words = sentence.split()
    vectors = [model_embedding.get_word_vector(word) for word in words]
    print(sentence)
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_dim)
    

# Import Data
df = pd.read_csv('df_queries.csv')
print('Data imported')

# Apply preprocessing function
punctuations = string.punctuation + "«»„“‚‘‘”’´`•"
stopwords = nltk.corpus.stopwords.words('german')
df['query_prep'] = df['query'].apply(preprocessing)
print('Data preprocessed')

# Apply Embedding using FastText pre-tained model 
# FastText Works with 157 languages and is trained on Common Crawl and Wikipedia.
# Link to download the pre-trained model on the german language: https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
model_path = "./cc.de.300.bin"  
model_embedding = fasttext.load_model(model_path)
model_words = model_embedding.get_words()                                                                         # Load the pre-trained model
vector_dim = 300
df['embedded_vectors'] = df['query_prep'].apply(lambda x: get_sentence_embedding(x, model_embedding, vector_dim)) # Apply the function to the "query" column of the DataFrame
print('Embedding process is complete')

# Apply PCA to reduce the size of vectors from 300 to 100
pca = PCA(n_components=100)
transformed_column = pca.fit_transform(df['embedded_vectors'].values.tolist()) # Apply PCA on the embedded vectors
df_pca = pd.DataFrame(data = transformed_column)                               # Convert the obtained vectors into a dataframe
combined = [list(row) for row in df_pca.to_numpy()]
df['embedded_vector_pca'] = combined                                           # Add new column containing the new vectors
print('PCA is applied')
pkl.dump(pca, open("pca.pkl","wb"))                                            # Load the PCA into pkl file

df2 = df[['query', 'embedded_vector_pca']]
df2.to_csv('new_data.csv', index=False, encoding='utf-8')
