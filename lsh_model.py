import pandas as pd
from annoy import AnnoyIndex


# Import data 
df2 = pd.read_csv('new_data.csv')

# Choose LSH parameters
num_trees = 150
num_neighbors = 10
vector_dim = 100

# Initialize LSH index
index = AnnoyIndex(vector_dim, metric='angular')
for i, row in df2[['embedded_vector_pca']].iterrows():
    vector = [float(x) for x in row.values[0][1:-1].split(',')]
    index.add_item(i, vector)

index.build(num_trees) 
index.save('annoy_model.ann') # Save the model