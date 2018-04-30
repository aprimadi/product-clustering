import json
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

with open('data/laptops.json') as json_data:
    d = json.load(json_data)
    products = [(row['name'].rstrip(), int(row['price'])) for row in d]

prices = [p[1] for p in products]
x = np.array(prices)

# costs = np.empty(10)
# costs[0] = None
# for k in range(1, 10):
#     km = KMeans(k)
#     km = km.fit(x.reshape(-1, 1))
#     costs[k] = km.score(x.reshape(-1, 1))
# plt.plot(costs)
# plt.title("Cost vs K")
# plt.show()

km = KMeans(8)
km = km.fit(x.reshape(-1, 1))

cluster_range = {}
for i in range(len(km.labels_)):
    c = km.labels_[i]
    price = prices[i]
    if c not in cluster_range:
        cluster_range[c] = { 'min': price, 'max': price }
    else:
        if price < cluster_range[c]['min']:
            cluster_range[c]['min'] = price
        if price > cluster_range[c]['max']:
            cluster_range[c]['max'] = price

sorted_range = sorted(cluster_range.items(), key=lambda x: x[1]['min'])
for item in sorted_range:
    print("{:,}".format(item[1]['min']), '-', "{:,}".format(item[1]['max']))
