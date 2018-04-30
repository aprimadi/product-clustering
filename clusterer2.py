import json
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer

lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
brands = set(w.rstrip() for w in open('brands.txt'))
types = set(w.rstrip() for w in open('types.txt'))

def tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens

def token_appraiser(t):
    if t in brands:
        return 10
    elif t in types:
        return 5
    else:
        return 1

def tokens_to_vector(tokens, word_index_map):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] += token_appraiser(t)
    return x

# Distance function is a squared distance between two vectors
def distance(u, v):
    diff = u - v
    return diff.dot(diff)

def cost(X, R, M):
    """TODO
    Args:
        X: Data - an (N, D) matrix
        R: Confidence matrix that a data entry belongs to cluster k ϵ K - (N, K)
        M: Cluster means - (K, D)
    """
    cost = 0
    for k in range(len(M)):
        # Method 1
        # for n in range(len(X)):
        #     cost += R[n,k]*d(M[k], X[n])

        # Method 2
        diff = X - M[k] # diff is an (N, D) matrix
        # sq_distances is an N-dimensional vector
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost

def kmeans(X, K, max_iter=100, beta=1.0):
    """TODO
    Args:
        X: Data
        K: Number of cluster
    Returns:
        A tuple M, R with the following definition:

        M: Mean points of the cluster
        R: Confidence score
    """
    N, D = X.shape
    M = np.zeros((K, D))
    # Each entry in R is like a confidence for i-th ϵ N dataset that it belongs
    # to cluster k ϵ K.
    #
    # N is the size of the data and K is the number of cluster.
    R = np.zeros((N, K))
    exponents = np.empty((N, K))

    # Initialize M to random
    for k in range(K):
        M[k] = X[np.random.choice(N)]

    costs = np.zeros(max_iter)
    for i in range(max_iter):
        if (i+1) % 10 == 0:
            print('Iteration:', i+1)
        # Step 1: determine assignments / responsibilities
        for k in range(K):
            for n in range(N):
                dist = distance(M[k], X[n])
                exponents[n,k] = np.exp(-beta * dist)

        # The second part of the term is a matrix of dimension (N, 1), numpy
        # will do an automatic stacking horizontally, turning the denominator
        # into an (N, K) matrix
        #
        # Note that we add one here to prevent division by zero
        R = exponents / (exponents.sum(axis=1, keepdims=True) + 1)

        # Step 2: recalculate means
        for k in range(K):
            # R[:,k] returns a matrix of size (1, N), when dotted with X (N, D),
            # it turns into (1, D) matrix.
            M[k] = R[:,k].dot(X) / R[:,k].sum()

        costs[i] = cost(X, R, M)
        if i > 0 and np.abs(costs[i] - costs[i-1]) < 10e-5:
            break

    return M, R

def print_clusters(R, names):
    # print out the clusters
    hard_responsibilities = np.argmax(R, axis=1) # is an N-size array of cluster identities
    # let's "reverse" the order so it's cluster identity -> word index
    cluster2word = {}
    for i in range(len(hard_responsibilities)):
        name = names[i]
        cluster = hard_responsibilities[i]
        if cluster not in cluster2word:
            cluster2word[cluster] = []
        cluster2word[cluster].append(name)

    # print out the words grouped by cluster
    for cluster, wordlist in cluster2word.items():
        print("cluster", cluster, "->", wordlist)
        print()
        print()

    print("Cluster size:", len(cluster2word))

def plot(X, K, R, index_word_map):
    N, D = X.shape
    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors) # colors is an (N, 3) matrix
    plt.figure(figsize=(80.0, 80.0))
    plt.scatter(X[:,0], X[:,1], s=N, alpha=0.9, c=colors)
    annotate(X, index_word_map)
    plt.savefig("test.png")

def annotate(X, index_word_map, eps=0.1):
    N, D = X.shape
    placed = np.empty((N, D))
    for i in range(N):
        # if x, y is too close to something already plotted, move it
        close = []

        x, y = X[i]
        for retry in range(3):
            for j in range(i):
                diff = np.array([x, y]) - placed[j]

                # if something is close, append it to the close list
                if diff.dot(diff) < eps:
                    close.append(placed[j])

            if close:
                # then the close list is not empty
                x += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
                y += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
                close = []
            else:
                # nothing close, let's break
                break

        placed[i] = (x, y)

        plt.annotate(
            s=index_word_map[i],
            xy=(X[i,0], X[i,1]),
            xytext=(x, y),
            arrowprops={
                'arrowstyle': '->',
                'color': 'black',
            }
        )


with open('data/laptops.json') as json_data:
    d = json.load(json_data)
    names = [row['name'].rstrip() for row in d]

word_index_map = {}
index_word_map = []
all_tokens = []
current_index = 0
for name in names:
    try:
        name = name.encode('ascii', 'ignore')
        name = name.decode('utf-8')
        tokens = tokenizer(name)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        print(e)

N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((N, D))
i = 0
for tokens in all_tokens:
    X[i, :] = tokens_to_vector(tokens, word_index_map)
    i += 1

transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

print('X shape:', X.shape)

# reducer = TSNE()
# Z = reducer.fit_transform(X)
# print('X shape:', X.shape)
# print('Z shape:', Z.shape)

M, R = kmeans(X, 70, max_iter=10)

# costs = np.empty(100)
# costs[0] = None
# for k in range(1, 10):
#     M, R = kmeans(X, k)
#     costs[k*10] = cost(X, R, M)
# plt.plot(costs)
# plt.title("Cost vs K")
# plt.show()

product_names = [" ".join(tokens) for tokens in all_tokens]
print_clusters(R, product_names)
