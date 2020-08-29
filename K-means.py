import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics
import tensorflow
import keras
import numpy

# print("K-means is real")

# loading data
digits = load_digits()

digits_data = scale(digits.data)  # as X before
Y = digits.target

k = len(numpy.unique(Y))  # dynamic k
# k=10 # static k

sample, features = digits_data.shape


# print(digits.target,k,digits_data.shape, sample, features)

# Score our model using a function from the sklearn website.
# It computes many different scores for different parts of our model.
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(Y, estimator.labels_),
             metrics.completeness_score(Y, estimator.labels_),
             metrics.v_measure_score(Y, estimator.labels_),
             metrics.adjusted_rand_score(Y, estimator.labels_),
             metrics.adjusted_mutual_info_score(Y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


my_first_kmeans = KMeans(n_clusters=k, init='random', n_init=10)  # we can set other params like max_iter, etc
my_second_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=30)  # parameters matter!

bench_k_means(my_first_kmeans, "First one", digits_data)
bench_k_means(my_second_kmeans, "Second one", digits_data)
