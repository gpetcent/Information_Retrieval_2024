import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import precision_at_k, mean_average_precision_at_k
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

# Get the current working directory
current_path = os.path.dirname(os.path.abspath(__file__))

# Load Last.FM
lastfm_data = pd.read_csv(r'C:\Users\kkafi\Desktop\Coding\Python\information_retrieval\Last.FM_Data\user_artists.dat', sep='\t')

# Load MovieLens 1M
movielens_data = pd.read_csv(r'C:\Users\kkafi\Desktop\Coding\Python\information_retrieval\Movielens_Data\ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')

# Create matrix for implicit feedback
def create_interaction_matrix(data, user_col, item_col, rating_col):
    interaction_matrix = data.groupby([user_col, item_col])[rating_col].sum().unstack().fillna(0)
    return interaction_matrix

lastfm_interactions = create_interaction_matrix(lastfm_data, 'userID', 'artistID', 'weight')
movielens_interactions = create_interaction_matrix(movielens_data, 'user_id', 'movie_id', 'rating')

# Turn into CSR format
lastfm_csr = sparse.csr_matrix(lastfm_interactions.values)
movielens_csr = sparse.csr_matrix(movielens_interactions.values)

def evaluate_model(model, train_data, test_data, K):
    model.fit(train_data.T)
    precision = precision_at_k(model, train_data.T, test_data.T, K)
    recall = mean_average_precision_at_k(model, train_data.T, test_data.T, K)
    return precision, recall

# Experiment Α
latent_factors = range(10, 101, 10)
iterations = 50
num_experiments = 10

precisions_lastfm = np.zeros((num_experiments, len(latent_factors)))
recalls_lastfm = np.zeros((num_experiments, len(latent_factors)))
precisions_movielens = np.zeros((num_experiments, len(latent_factors)))
recalls_movielens = np.zeros((num_experiments, len(latent_factors)))

for experiment in range(num_experiments):
    train_lastfm, test_lastfm = train_test_split(lastfm_csr, test_size=0.2, random_state=experiment)
    train_movielens, test_movielens = train_test_split(movielens_csr, test_size=0.2, random_state=experiment)

    for i, factors in enumerate(latent_factors):
        model = BayesianPersonalizedRanking(factors=factors, iterations=iterations)
        precision, recall = evaluate_model(model, train_lastfm, test_lastfm, 10)
        precisions_lastfm[experiment, i] = precision
        recalls_lastfm[experiment, i] = recall
        
        model = BayesianPersonalizedRanking(factors=factors, iterations=iterations)
        precision, recall = evaluate_model(model, train_movielens, test_movielens, 10)
        precisions_movielens[experiment, i] = precision
        recalls_movielens[experiment, i] = recall

# Compute means and standard deviations
mean_precisions_lastfm = precisions_lastfm.mean(axis=0)
std_precisions_lastfm = precisions_lastfm.std(axis=0)
mean_recalls_lastfm = recalls_lastfm.mean(axis=0)
std_recalls_lastfm = recalls_lastfm.std(axis=0)

mean_precisions_movielens = precisions_movielens.mean(axis=0)
std_precisions_movielens = precisions_movielens.std(axis=0)
mean_recalls_movielens = recalls_movielens.mean(axis=0)
std_recalls_movielens = recalls_movielens.std(axis=0)

# Plotting results for Last.FM
plt.figure()
plt.plot(latent_factors, mean_precisions_lastfm, label='Precision')
plt.fill_between(latent_factors, mean_precisions_lastfm - std_precisions_lastfm, mean_precisions_lastfm + std_precisions_lastfm, color='b', alpha=0.2)
plt.xlabel('Latent Factors')
plt.ylabel('Score')
plt.legend()
plt.savefig('lastfm_lfp.pdf')

plt.figure()
plt.plot(latent_factors, mean_recalls_lastfm, label='Recall')
plt.fill_between(latent_factors, mean_recalls_lastfm - std_recalls_lastfm, mean_recalls_lastfm + std_recalls_lastfm, color='b', alpha=0.2)
plt.xlabel('Latent Factors')
plt.ylabel('Score')
plt.legend()
plt.savefig('lastfm_lfr.pdf')

# Plotting results for MovieLens 1M
plt.figure()
plt.plot(latent_factors, mean_precisions_movielens, label='Precision')
plt.fill_between(latent_factors, mean_precisions_movielens - std_precisions_movielens, mean_precisions_movielens + std_precisions_movielens, color='b', alpha=0.2)
plt.xlabel('Latent Factors')
plt.ylabel('Score')
plt.legend()
plt.savefig('movielens_lfp.pdf')

plt.figure()
plt.plot(latent_factors, mean_recalls_movielens, label='Recall')
plt.fill_between(latent_factors, mean_recalls_movielens - std_recalls_movielens, mean_recalls_movielens + std_recalls_movielens, color='b', alpha=0.2)
plt.xlabel('Latent Factors')
plt.ylabel('Score')
plt.legend()
plt.savefig('movielens_lfr.pdf')

# Experiment Β
ks = range(2, 21, 2)
factors = 50

precisions_lastfm = np.zeros((num_experiments, len(ks)))
recalls_lastfm = np.zeros((num_experiments, len(ks)))
precisions_movielens = np.zeros((num_experiments, len(ks)))
recalls_movielens = np.zeros((num_experiments, len(ks)))

for experiment in range(num_experiments):
    train_lastfm, test_lastfm = train_test_split(lastfm_csr, test_size=0.2, random_state=experiment)
    train_movielens, test_movielens = train_test_split(movielens_csr, test_size=0.2, random_state=experiment)

    for i, k in enumerate(ks):
        model = BayesianPersonalizedRanking(factors=factors, iterations=iterations)
        precision, recall = evaluate_model(model, train_lastfm, test_lastfm, k)
        precisions_lastfm[experiment, i] = precision
        recalls_lastfm[experiment, i] = recall
        
        model = BayesianPersonalizedRanking(factors=factors, iterations=iterations)
        precision, recall = evaluate_model(model, train_movielens, test_movielens, k)
        precisions_movielens[experiment, i] = precision
        recalls_movielens[experiment, i] = recall

# Compute means and standard deviations
mean_precisions_lastfm = precisions_lastfm.mean(axis=0)
std_precisions_lastfm = precisions_lastfm.std(axis=0)
mean_recalls_lastfm = recalls_lastfm.mean(axis=0)
std_recalls_lastfm = recalls_lastfm.std(axis=0)

mean_precisions_movielens = precisions_movielens.mean(axis=0)
std_precisions_movielens = precisions_movielens.std(axis=0)
mean_recalls_movielens = recalls_movielens.mean(axis=0)
std_recalls_movielens = recalls_movielens.std(axis=0)

# Plotting results for Last.FM
plt.figure()
plt.plot(ks, mean_precisions_lastfm, label='Precision')
plt.fill_between(ks, mean_precisions_lastfm - std_precisions_lastfm, mean_precisions_lastfm + std_precisions_lastfm, color='b', alpha=0.2)
plt.xlabel('Top-K')
plt.ylabel('Score')
plt.legend()
plt.savefig('lastfm_tkp.pdf')

plt.figure()
plt.plot(ks, mean_recalls_lastfm, label='Recall')
plt.fill_between(ks, mean_recalls_lastfm - std_recalls_lastfm, mean_recalls_lastfm + std_recalls_lastfm, color='b', alpha=0.2)
plt.xlabel('Top-K')
plt.ylabel('Score')
plt.legend()
plt.savefig('lastfm_tkr.pdf')

# Plotting results for MovieLens 1M
plt.figure()
plt.plot(ks, mean_precisions_movielens, label='Precision')
plt.fill_between(ks, mean_precisions_movielens - std_precisions_movielens, mean_precisions_movielens + std_precisions_movielens, color='b', alpha=0.2)
plt.xlabel('Top-K')
plt.ylabel('Score')
plt.legend()
plt.savefig('movielens_tkp.pdf')

plt.figure()
plt.plot(ks, mean_recalls_movielens, label='Recall')
plt.fill_between(ks, mean_recalls_movielens - std_recalls_movielens, mean_recalls_movielens + std_recalls_movielens, color='b', alpha=0.2)
plt.xlabel('Top-K')
plt.ylabel('Score')
plt.legend()
plt.savefig('movielens_tkr.pdf')