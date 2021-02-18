


"""
General idea: vectorize the text. then perform clustering

To see similarity of a new text to a more common one, use 
same clustering algorithm

Another idea... check to see the severity of your data stolen


both require vectorization of the text

"""

from sklearn.cluster import KMeans
import numpy as np

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer

import os





############## example of fitting then predicting ###########
# vectors = np.array([
#     [1,1,1,1],
#     [2,2,2,2],
#     [8,8,8,8],
#     [9,9,9,9],
# ])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(vectors)
# print(kmeans.labels_)
# print(kmeans.predict([[-1,-1,-1,-1]]))
# print(kmeans.cluster_centers_)
#############################################################




def data_counts(lemmatized):
    num_data = 0
    for word in lemmatized:
        if word == 'data':
            num_data += 1

    return num_data





def vectorize(filename):
    vector = []
    with open(filename) as f:
        current_text_lines = []
        line = f.readline()
        while line:
            current_text_lines.append(line)
            line = f.readline()

    current_text = ' '.join(current_text_lines)

    current_text_tokens = word_tokenize(current_text)

    word_lem = WordNetLemmatizer()

    lemmatized = [word_lem.lemmatize(token) for token in current_text_tokens]

    vector.append(data_counts(lemmatized))

    return vector




def vectorize_directory(directory):
    vector_dictionary = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            vector_dictionary[filename] = vectorize(os.path.join(directory, filename))

    filenames = []
    vectors = []

    for filename in vector_dictionary:
        filenames.append(filename)
        vectors.append(vector_dictionary[filename])

    return filenames, vectors




def cluster_terms(directory, n_clusters=2):
    filenames, vectors = vectorize_directory(directory)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    print(kmeans.labels_)

    for i in range(len(filenames)):
        print(filenames[i], ':', kmeans.labels_[i])

    return filenames, vectors


if __name__ == '__main__':
    # vectorize('data/facebook_terms_and_data_policy.txt')
    # vectorize_directory('data')
    cluster_terms('data')


