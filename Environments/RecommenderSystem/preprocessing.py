import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings; warnings.simplefilter('ignore')


input_file = 'movies.csv'
df = pd.read_csv(input_file)
df['state_descriptor'] = df['genres']
tf = TfidfVectorizer(analyzer='word',min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['state_descriptor'])
tfidf_matrix_dense = tfidf_matrix.todense()
tfidf_matrix_dense = np.unique(tfidf_matrix_dense,axis=0)
tfidf_matrix_numpy = np.zeros((tfidf_matrix_dense.shape[0],tfidf_matrix_dense.shape[1]+1))
tfidf_matrix_numpy[:,:-1] = np.asarray(tfidf_matrix_dense)
tfidf_matrix_numpy[0:int(np.round((tfidf_matrix_numpy.shape[0]-1)*0.6)),tfidf_matrix_numpy.shape[1]-1] = 1
tfidf_matrix_numpy[int(np.round((tfidf_matrix_numpy.shape[0]-1)*0.6)):int(np.round((tfidf_matrix_numpy.shape[0]-1)*0.9)),tfidf_matrix_numpy.shape[1]-1] = 10
tfidf_matrix_numpy[int(np.round((tfidf_matrix_numpy.shape[0]-1)*0.9)):,tfidf_matrix_numpy.shape[1]-1] = 30
rewards_vector = tfidf_matrix_numpy[:,tfidf_matrix_numpy.shape[1]-1]

# Create array wit hadditional entries for id and distance metric
tfidf_matrix_sorted = np.zeros((tfidf_matrix_numpy.shape[0],tfidf_matrix_numpy.shape[1]))
tfidf_matrix_sorted[:,:] = tfidf_matrix_numpy[:,:]

for i in range(0,tfidf_matrix_sorted.shape[0]):
    help_array = np.zeros((tfidf_matrix_numpy.shape[0]-(i+1), tfidf_matrix_numpy.shape[1] + 1))
    distance_metric = np.sum(abs(tfidf_matrix_sorted[i,:-1] - tfidf_matrix_sorted[i+1:,:-1]),axis=1) # here with manhattan distance
    help_array[:,0] = distance_metric[:]
    help_array[:,1:] = tfidf_matrix_sorted[i+1:,:]
    help_array = help_array[help_array[:, 0].argsort()]
    tfidf_matrix_sorted[i+1:,:] = help_array[:,1:]
rewards_vector_sorted = tfidf_matrix_sorted[:,tfidf_matrix_sorted.shape[1]-1]


with open('tf_idf_matrix.npy','wb') as f:
    np.save(f,tfidf_matrix_dense)
with open('rewards_vector.npy','wb') as f:
    np.save(f,rewards_vector)
