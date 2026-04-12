import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from game import simulate_equity_batch
from config import Config
import eval7

def simulate_features(num_sims=None):
    num_sims = num_sims or Config.NUM_SIMS
    feature_batch_size = max(64, min(Config.BATCH_SIZE, 1024))
    batches = (num_sims + feature_batch_size - 1) // feature_batch_size
    features = np.zeros((num_sims, 5))  # equity, suited, rank_sum, pot_odds, position
    for i in range(batches):
        start, end = i * feature_batch_size, min((i + 1) * feature_batch_size, num_sims)
        batch_decks = [eval7.Deck() for _ in range(end - start)]
        for deck in batch_decks:
            deck.shuffle()
        batch_holes = [deck.deal(2) for deck in batch_decks]
        batch_boards = [[] for _ in range(end - start)]
        equities = simulate_equity_batch(batch_holes, batch_boards, num_opponents=Config.NUM_OPPONENTS)
        for j, (hole, equity) in enumerate(zip(batch_holes, equities)):
            idx = i * feature_batch_size + j
            suited = int(hole[0].suit == hole[1].suit)
            rank_sum = ((hole[0].rank + 2) + (hole[1].rank + 2)) / 28.0  # Adjust for eval7 rank 0-12 -> equiv 2-14
            pot_odds = np.random.uniform(0.1, 0.9)
            position = np.random.randint(0, 6) / 5.0
            features[idx] = [equity, suited, rank_sum, pot_odds, position]
    return features

def create_buckets(features, num_buckets=Config.NUM_BUCKETS):
    # Cap clusters to sample count so quick smoke tests with fewer simulations still run.
    num_buckets = min(num_buckets, len(features))
    if len(features) > 8192:
        kmeans = MiniBatchKMeans(n_clusters=num_buckets, random_state=42, batch_size=min(4096, len(features))).fit(features)
    else:
        kmeans = KMeans(n_clusters=num_buckets, random_state=42).fit(features)
    return kmeans.labels_, kmeans.cluster_centers_

if __name__ == "__main__":
    features = simulate_features()
    buckets, centroids = create_buckets(features)
    np.save('buckets.npy', buckets)
    np.save('centroids.npy', centroids)
    print(f"Buckets created: {len(set(buckets))} unique")