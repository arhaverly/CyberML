


"""
General idea: vectorize the text. then perform clustering

To see similarity of a new text to a more common one, use 
same clustering algorithm

Another idea... check to see the severity of your data stolen


both require vectorization of the text

"""



'''
Overall design:
1. Create a feature vector using NLP
2. Perform clustering
    1. K-means, Latent Dirichlet Allocation (LDA), or another algorithm

'''




from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import numpy as np

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.util import bigrams, trigrams, ngrams

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering



import os
import seaborn as sns
import matplotlib.pyplot as plt
from random import random



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
#############################################################`




def data_counts(lemmatized):
    num_data = 0
    for word in lemmatized:
        if word == 'data':
            num_data += 1

    return num_data




# https://www.researchgate.net/publication/325431586_Measuring_Similarity_among_Legal_Court_Case_Documents


# TF-IDF is used to measure similarities between two documents. Can create a confusion matrix
# Then rank most similar ToS to least similar ToS
def tf_idf_vectorization(directory, vectors):
    # print(text.ENGLISH_STOP_WORDS)

    document_ids = [(i, os.path.join(directory, f)) for i, f in enumerate(os.listdir(directory))]
    # print(document_ids)
    # print(len(document_ids))

    documents = [' '.join(read_and_clean(f)) for i, f in document_ids]
    tf = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS, max_features=30)

    tfidf = tf.fit_transform(documents)

    feature_names = tf.get_feature_names()

    feature_index = tfidf[0,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf[0,x] for x in feature_index])

    # for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
    #     print(w + ',', end=' ')
    # print('\n\n\n')


    for i, vector in enumerate(vectors):
        tf_idf_vector = np.array(tfidf[i].toarray()[0])
        for tf in tf_idf_vector:
            vectors[i].append(tf)




# vectorize words directly. embed two text portions of the same section and compare them



# Doc2vec can be used to vectorize an entire document







# can use the whole document, just a summary, or both














# maybe do this
# find the table of contents, compare the lemmatizations directly





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

    # vector.append(data_counts(lemmatized))

    return vector




def vectorize_directory(directory):
    vector_dictionary = {}
    for filename in os.listdir(directory):
        vector_dictionary[filename] = vectorize(os.path.join(directory, filename))

    filenames = []
    vectors = []

    for filename in vector_dictionary:
        # print(filename, ':', vector_dictionary[filename])
        filenames.append(filename)
        vectors.append(vector_dictionary[filename])


    tf_idf_vectorization(directory, vectors)
    # print(vectors)

    return filenames, vectors



def normalize_array(arr):
    minimum = 10000000
    maximum = -10000000
    for elem in arr:
        if elem < minimum:
            minimum = elem
        if elem > maximum:
            maximum = elem

    maximum -= minimum

    for i in range(len(arr)):
        arr[i] = (arr[i]-minimum) / maximum


def cluster_terms(directory, n_clusters=2):
    filenames, vectors = vectorize_directory(directory)


    n_clusters_range = range(2,100)
    fig, axs = plt.subplots(2)
    fig.suptitle('Different Clustering Algorithms with Validity Indices')


    # KMeans
    kmeans_validity_indices = {'Davies Bouldin Score': [],
                        'Calinski Harabasz Score':[],
                        'Silhouette Score':[],
                        }

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
        labels = kmeans.labels_

        # the closer to zero, the better the clustering
        kmeans_validity_indices['Davies Bouldin Score'].append(davies_bouldin_score(vectors, labels))
        kmeans_validity_indices['Calinski Harabasz Score'].append(calinski_harabasz_score(vectors, labels))
        kmeans_validity_indices['Silhouette Score'].append(silhouette_score(vectors, labels))



    for key in kmeans_validity_indices.keys():
        normalize_array(kmeans_validity_indices[key])


    for i in range(len(filenames)):
        # print(filenames[i], ':', kmeans.labels_[i])
        pass

    for key in kmeans_validity_indices.keys():
        axs[0].plot(n_clusters_range, kmeans_validity_indices[key], label=key)

    axs[0].set_title('KMeans')


    # Spectral Clustering
    spectral_validity_indices = {'Davies Bouldin Score': [],
                        'Calinski Harabasz Score':[],
                        'Silhouette Score':[],
                        }

    for n_clusters in n_clusters_range:
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=0, assign_labels='discretize').fit(vectors)
        labels = spectral.labels_

        # the closer to zero, the better the clustering
        spectral_validity_indices['Davies Bouldin Score'].append(davies_bouldin_score(vectors, labels))
        spectral_validity_indices['Calinski Harabasz Score'].append(calinski_harabasz_score(vectors, labels))
        spectral_validity_indices['Silhouette Score'].append(silhouette_score(vectors, labels))



    for key in spectral_validity_indices.keys():
        normalize_array(spectral_validity_indices[key])


    kmeans = KMeans(n_clusters=50, random_state=0).fit(vectors)
    labels = kmeans.labels_
    for i in range(len(filenames)):
        print(filenames[i], ':', kmeans.labels_[i], end=', ')
        pass

    for key in spectral_validity_indices.keys():
        axs[1].plot(n_clusters_range, spectral_validity_indices[key], label=key)


    axs[1].set_title('Spectral Clustering')




    plt.xlabel('Number of Clusters')
    plt.ylabel('Validity Index Score')

    axs[0].legend()
    axs[1].legend()

    plt.savefig('validity_indices.png')
    plt.show()


    return filenames, vectors



def get_table_contents(filename):
    # requires #START_TABLE_OF_CONTENTS# and #END_TABLE_OF_CONTENTS# in the file
    table_of_contents = []
    with open(filename) as f:
        line = True
        in_table = False
        while line:
            line = f.readline()
            if '#START_TABLE_OF_CONTENTS#' in line:
                in_table = True
                continue

            
            elif '#END_TABLE_OF_CONTENTS#' in line:
                break

            elif in_table:
                table_of_contents.append(line.replace('\n', ''))

    # print('Table of contents:', table_of_contents)
    return table_of_contents


def find_collect_section_name(table_of_contents):
    collect_section = False
    next_section = False
    for section in table_of_contents:
        if collect_section != False:
            next_section = section
            break
        if 'collect' in section:
            collect_section = section
        

    if collect_section == False:
        raise Exception("Didn't find collection section")

    return collect_section, next_section


def get_collect_section(filename, collect_section, next_section):
    with open(filename) as f:
        line = True
        in_section = 0
        section_text = ''
        line_counter = 0
        while line:
            line_counter += 1 
            line = f.readline()
            if collect_section in line:
                in_section += 1 # need to wait for the second title


            if in_section == 2:
                if next_section in line:
                    return section_text

                section_text += line



def get_lists(section_text):
    # split into array of lines
    lines = section_text.split('\n')
    # print(lines)

    items_from_lists = []

    # find all tabbed-in items. take words before punctuation
    for line in lines:
        if line.startswith('    '):
            # print(line)
            item = ''
            for char in line[4:]:
                if char in ['.', ':']:
                    break
                
                item += char

            items_from_lists.append(item)


    # Am I on the right track?
    for line in lines:
        sentences = line.split('.')
        for sentence in sentences:
            if ',' in sentence:
                list_items = sentence.split(',')
                if len(list_items) > 2:
                    for item in list_items:
                        if len(item.split()) < 6:
                            items_from_lists.append(item)


    print(items_from_lists)



def read_and_clean(filename):
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

    punctuation = [',', '.', ';', ':', '?']

    cleaned = [word for word in lemmatized if word not in punctuation]

    return cleaned



def find_near_keyword(keyword, word_list, distance):
    positions = []
    for i, word in enumerate(word_list):
        if word == keyword:
            positions.append(i)

    nearby_words = []
    for position in positions:
        nearby_words += word_list[position-distance: position+distance]


    # print(nearby_words)
    return nearby_words



def get_document_word_freqs(filename):
    cleaned = read_and_clean(filename)
    total_words = len(cleaned)
    word_freqs = {}
    for word in cleaned:
        if word in word_freqs:
            word_freqs[word] += 1/total_words
        else:
            word_freqs[word] = 1/total_words

    return word_freqs


def rank_words(nearby_words, document_word_freqs):
    set_words = set(nearby_words)
    # set_words = set([])
    # for word in nearby_words:
    #     set_words.add(word)

    words_with_ranks = []
    for word in set_words:
        if word in document_word_freqs:
            rank = 1/document_word_freqs[word]
        else:
            rank = 100
        word_with_rank = (word, rank)
        words_with_ranks.append(word_with_rank)


    words_with_ranks.sort(key=lambda x: x[1], reverse=True)
    return words_with_ranks




def compare_new_to_known_tos(new, known, directory):
    filenames, vectors = vectorize_directory(directory)
    new_vector = vectors[filenames.index(new)]
    # print('new:', new_vector)

    known_vectors = [vectors[filenames.index(os.path.basename(tos))] for tos in known]
    # print('known:', known_vectors)

    print('Cosine Similarity')
    cosine_similarities = [np.sum(np.multiply(np.array(new_vector), np.array(known_vector))) for known_vector in known_vectors]
    # print('cosine_similarities', cosine_similarities)

    name_with_similarity = [(known[i], cosine_similarities[i]) for i in range(len(known))]

    # print(name_with_similarity)

    name_with_similarity.sort(key=lambda x: x[1], reverse=True)

    print(name_with_similarity, '\n')



    print('K-Means')
    kmeans = KMeans(n_clusters=len(known), random_state=0).fit(known_vectors)
    # print(kmeans.predict([new_vector]))
    print(known[list(kmeans.labels_).index(kmeans.predict([new_vector])[0])])


    print('Spectral Clustering')
    spectral = SpectralClustering(n_clusters=len(known), random_state=0, assign_labels='discretize').fit_predict(known_vectors, new_vector)
    print(spectral)
    # print(kmeans.predict([new_vector]))
    print(known[list(spectral.labels_).index(spectral.fit_predict([new_vector])[0])])





def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    print(counts)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)




def hierarchical_cluster(filenames, vectors):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(vectors)

    plot_dendrogram(model, truncate_mode='level', p=10)
    plt.show()




if __name__ == '__main__':
    # 1 - Create feature vector, cluster
    # vectorize('data/facebook_terms_and_data_policy.txt')
    # vectorize_directory('data')
    # cluster_terms('data/all_terms', n_clusters=5)


    # known_tos = [os.path.join('data/all_terms', tos) for tos in ['activision', 'adobe.txt', 'adp', 'android', 'audible', 'babbel', 'bumble']]
    # new_tos = 'uber'
    # compare_new_to_known_tos(new_tos, known_tos, 'data/all_terms')











    ############################## Preliminary Results ##############################

    # Break down this hypothesis: A clustering algorithm can identify which known ToS is most similar to a novel ToS.
    # Hypothesis #1: A clustering algorithm can be used to identify which known ToS is most similar to a novel ToS.
    # Hypothesis #2: A vector similarity measure can be used to compare the similarity of one ToS to another ToS.
    # Hypothesis #3: A hierarchical clustering algorithm can be used to visually examine the similarity between many ToS.


    filenames, vectors = vectorize_directory('data/all_terms')




    print('Hierarchical Cluster')
    # perform the hierarchical cluster with plotting
    hierarchical_cluster(filenames, vectors)










    print('\n\n\nSimilarity indices')
    # how close are they?


    # how are they similar? what features?
    # what component leads the most to this closeness
        # whjich of the top words score leads the most to similarity





    print('\n\n\nHow well can clustering describe similarity?')

























































    # tf_idf_vectorization('data', [[] for _ in range(100)])








    # 2 - get lists of nouns
    # table_of_contents = get_table_contents('data/facebook_with_table_contents.txt')
    # collect_section_name, next_section_name = find_collect_section_name(table_of_contents)
    # collect_section_text = get_collect_section('data/facebook_with_table_contents.txt', collect_section_name, next_section_name)
    # lists = get_lists(collect_section_text)


    # 3 - Find keyword, use NLP to derive new keyword
    # cleaned = read_and_clean('data/facebook_terms_and_data_policy.txt')
    # # print(cleaned)
    # nearby_words = find_near_keyword('browser', cleaned, 10)
    # print('\nnearby words:\n', nearby_words)

    # bigrams = list(bigrams(nearby_words))
    # print('\nbigrams\n', bigrams)


    # document_word_freqs = get_document_word_freqs('computer_networks.txt')

    # ranked_words = rank_words(nearby_words, document_word_freqs)
    # print('\nranked words:\n', ranked_words)
















# What kind of outputs / label can i get out
# May ened to manually label
# Use NLP to extract feature vectors
'''
Superfised vs unsupervised
label terms of services with keywords? may be too much

transfer learning





z



winner: extract feature vectors using NLP
Unsupervised Similarity between Terms of Services using NLP




Come up with comments/critiques for other students projects
'''

















# ducky
'''
method to get "what":
1. Break into sections
2. Find section to determine "what" data is being collected
    if it has "collect" in the title

3. Figure out "what" data is being collected
    1. Find lists (tabbed in first words before punctuation or sentence with 2 or more commas)
    2. Get items from the list
    3. Anything else?




I extracted the lists of information that is being collected.


Now I'm looking into the "how". To a computer person, it's quite easy to understand how they are collecting it (cookies, for example). Do you think a good path forward would be to search the web and use NLP to extract how we think the data is being collected?




'''









'''
1. where/who they share the data with




2. how long they keep the data






3. Keyword search and derive a new keyword based on what you read
    a. find where the keywords are
    b. Use NLP to determine words that are similar


example_text =  '...Information we obtain from these devices includes:
                Device attributes: information such as the operating system,
                hardware and software versions, battery level, signal strength,
                available storage space, browser type, app and file names and types,
                and plugins...'

key = 'browser'

finds the position, reads all words in the sentence



'''




















