import json
import nltk
import hdbscan
import math
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer

lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt') if not w.startswith('#'))
nontypes = set(w.rstrip() for w in open('nontypes.txt') if not w.startswith('#'))
brands = set(w.rstrip() for w in open('brands.txt') if not w.startswith('#'))
subbrands = set(w.rstrip() for w in open('subbrands.txt') if not w.startswith('#'))
versions = set(w.rstrip() for w in open('versions.txt') if not w.startswith('#'))
wordmap = {}
for line in open('wordmap.txt'):
    if line.strip() != "":
        tokens = line.split()
        wordmap[tokens[0]] = tokens[1]

def tokenizer(s):
    tokenizer = RegexpTokenizer(r'\/')

    s = s.lower()
    tokens = [w2 for w in nltk.tokenize.word_tokenize(s) for w1 in w.split('/') for w2 in w1.split('-')]
    tokens = [t for t in tokens if len(t) > 2 or t == 'hp'] # remove short words, they're probably not useful
    tokens = [lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [wordmap.get(t, t) for t in tokens] # apply word mapping
    return tokens

def token_appraiser(t):
    if t in nontypes:
        return -100
    elif t in brands:
        return 20
    elif t in subbrands:
        return 10
    elif t in versions:
        return 5
    else:
        return 1

def tokens_to_vector(tokens, word_index_map):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] += token_appraiser(t)
    return x

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
    products = [(row['name'].rstrip(), int(row['price'])) for row in d if int(row['price']) > 2000000]

word_index_map = {}
index_word_map = []
all_tokens = []
current_index = 0
for product in products:
    try:
        name = product[0].encode('ascii', 'ignore')
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

# Stack a price vector
pv = np.array([product[1] / 1000000 for product in products])
X = np.column_stack((X, pv))
print('X shape:', X.shape)

clusterer = hdbscan.HDBSCAN()
clusterer.fit(X)

clusters = {}
for i in range(len(clusterer.labels_)):
    c = clusterer.labels_[i]
    if c not in clusters:
        clusters[c] = []
    clusters[c].append({ 'name': " ".join(all_tokens[i]), 'price': products[i][1], 'tokens': all_tokens[i] })
sorted_clusters = sorted(clusters.items(), key=lambda x: x[0])
for item in sorted_clusters:
    min_price = math.inf
    max_price = 0
    for product in item[1]:
        min_price = min(min_price, product['price'])
        max_price = max(max_price, product['price'])

    if min_price * 3 < max_price:
        # Non-successful cluster
        print('Cluster', item[0], ' -> (', "{:,}".format(min_price), '-', "{:,}".format(max_price), ')')
        for product in item[1]:
            print("\t", product['name'], '-', "{:,}".format(product['price']))
    else:
        product_descs = []
        for product in item[1]:
            for t in product['tokens']:
                if t not in product_descs and (t in brands or t in subbrands or t in versions):
                    product_descs.append(t)
        print('Cluster', item[0], '->', ' '.join(product_descs), '(', "{:,}".format(min_price), '-', "{:,}".format(max_price), ')')
        if len(product_descs) == 0:
            for product in item[1]:
                print("\t", product['name'], '-', "{:,}".format(product['price']))
