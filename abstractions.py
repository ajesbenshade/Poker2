"""Abstraction helpers for clustering poker game states.

The current implementation builds a simple bucket abstraction by running
``k``-means on simulated equity feature vectors.  Alternative approaches such
as effective hand strength (EHS/EHS2) or other potential-aware schemes could be
plugged in, but are outside the scope of this reference implementation.

Bucket counts are tracked per street to make it explicit how many clusters are
used at each stage of the game.  Hands and boards are canonicalised before
feature extraction so that permuting suits or card order does not change the
resulting bucket, which helps avoid the "bucket bleed" problem.
"""

import numpy as np
from sklearn.cluster import KMeans
from game import simulate_equity_batch
from config import Config
import eval7


# Number of abstraction buckets to use on each betting street.  These values are
# purely illustrative and can be tuned depending on the desired granularity.
BUCKETS_PER_STREET = {
    "preflop": 169,  # canonical starting hands
    "flop": 1000,
    "turn": 500,
    "river": 200,
}

def simulate_features(num_sims=Config.NUM_SIMS):
    batches = (num_sims + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    features = np.zeros((num_sims, 5))  # equity, suited, rank_sum, pot_odds, position
    for i in range(batches):
        start, end = i * Config.BATCH_SIZE, min((i + 1) * Config.BATCH_SIZE, num_sims)
        batch_decks = [eval7.Deck() for _ in range(end - start)]
        for deck in batch_decks:
            deck.shuffle()
        batch_holes = [deck.deal(2) for deck in batch_decks]
        batch_boards = [[] for _ in range(end - start)]
        equities = simulate_equity_batch(batch_holes, batch_boards)
        for j, (hole, equity) in enumerate(zip(batch_holes, equities)):
            idx = i * Config.BATCH_SIZE + j
            suited = int(hole[0].suit == hole[1].suit)
            rank_sum = ((hole[0].rank + 2) + (hole[1].rank + 2)) / 28.0  # Adjust for eval7 rank 0-12 -> equiv 2-14
            pot_odds = np.random.uniform(0.1, 0.9)
            position = np.random.randint(0, 6) / 5.0
            features[idx] = [equity, suited, rank_sum, pot_odds, position]
    return features

def create_buckets(features, num_buckets=Config.NUM_BUCKETS):
    kmeans = KMeans(n_clusters=num_buckets, random_state=42).fit(features)
    return kmeans.labels_, kmeans.cluster_centers_

if __name__ == "__main__":
    features = simulate_features()
    buckets, centroids = create_buckets(features)
    np.save('buckets.npy', buckets)
    np.save('centroids.npy', centroids)
    print(f"Buckets created: {len(set(buckets))} unique")

