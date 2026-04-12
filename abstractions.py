import numpy as np
import torch
from sklearn.cluster import KMeans

from game import simulate_equity_batch
from config import Config
from datatypes import Infoset, Street

try:
    import eval7
except ImportError:
    eval7 = None
    from treys import Card as TreysCard, Deck as TreysDeck


def _new_deck():
    return eval7.Deck() if eval7 is not None else TreysDeck()


def _deal_cards(deck, count):
    if eval7 is not None:
        return deck.deal(count)
    drawn = deck.draw(count)
    return drawn if isinstance(drawn, list) else [drawn]


def _hole_features(hole):
    if eval7 is not None:
        suited = int(hole[0].suit == hole[1].suit)
        rank_sum = ((hole[0].rank + 2) + (hole[1].rank + 2)) / 28.0
        return suited, rank_sum

    hole_strings = [TreysCard.int_to_str(card) for card in hole]
    suited = int(hole_strings[0][1].lower() == hole_strings[1][1].lower())
    rank_sum = sum("23456789TJQKA".index(card_str[0]) + 2 for card_str in hole_strings) / 28.0
    return suited, rank_sum


def simulate_features(num_sims=None):
    num_sims = Config.NUM_SIMS if num_sims is None else num_sims
    batches = (num_sims + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    features = np.zeros((num_sims, 5))  # equity, suited, rank_sum, pot_odds, position
    for i in range(batches):
        start, end = i * Config.BATCH_SIZE, min((i + 1) * Config.BATCH_SIZE, num_sims)
        batch_decks = [_new_deck() for _ in range(end - start)]
        for deck in batch_decks:
            deck.shuffle()
        batch_holes = [_deal_cards(deck, 2) for deck in batch_decks]
        batch_boards = [[] for _ in range(end - start)]
        equities = simulate_equity_batch(batch_holes, batch_boards)
        for j, (hole, equity) in enumerate(zip(batch_holes, equities)):
            idx = i * Config.BATCH_SIZE + j
            suited, rank_sum = _hole_features(hole)
            pot_odds = np.random.uniform(0.1, 0.9)
            position = np.random.randint(0, 6) / 5.0
            features[idx] = [equity, suited, rank_sum, pot_odds, position]
    return features

def create_buckets(features, num_buckets=None):
    num_buckets = Config.NUM_BUCKETS if num_buckets is None else num_buckets
    kmeans = KMeans(n_clusters=num_buckets, random_state=42).fit(features)
    return kmeans.labels_, kmeans.cluster_centers_


def feature_vector_size(history_length=None):
    history_length = Config.HISTORY_FEATURES if history_length is None else history_length
    return (Config.CARD_FEATURES * 2) + len(Street) + 8 + history_length


def encode_infoset(infoset: Infoset, history_length=None):
    history_length = Config.HISTORY_FEATURES if history_length is None else history_length
    private_cards = np.zeros(Config.CARD_FEATURES, dtype=np.float32)
    board_cards = np.zeros(Config.CARD_FEATURES, dtype=np.float32)
    street_features = np.zeros(len(Street), dtype=np.float32)
    history_features = np.full(history_length, -1.0, dtype=np.float32)

    for card_id in infoset.private_cards:
        if 0 <= card_id < Config.CARD_FEATURES:
            private_cards[card_id] = 1.0
    for card_id in infoset.board_cards:
        if 0 <= card_id < Config.CARD_FEATURES:
            board_cards[card_id] = 1.0

    street_features[infoset.street.value] = 1.0

    trimmed_history = infoset.history[-history_length:]
    if trimmed_history:
        action_scale = max(Config.NUM_ACTIONS - 1, 1)
        history_features[-len(trimmed_history):] = np.asarray(trimmed_history, dtype=np.float32) / action_scale

    stack_scale = max(Config.INITIAL_STACK, 1.0)
    bucket_scale = max(Config.NUM_BUCKETS - 1, 1)
    scalar_features = np.asarray([
        infoset.bucket_id / bucket_scale,
        float(infoset.acting_player),
        infoset.pot_size / stack_scale,
        infoset.current_bet / stack_scale,
        infoset.effective_stack / stack_scale,
        infoset.stack_sizes[0] / stack_scale,
        infoset.stack_sizes[1] / stack_scale,
        infoset.history_length / max(history_length, 1),
    ], dtype=np.float32)

    return np.concatenate((
        private_cards,
        board_cards,
        street_features,
        scalar_features,
        history_features,
    )).astype(np.float32, copy=False)


def encode_infosets(infosets, device=Config.DEVICE, history_length=None):
    history_length = Config.HISTORY_FEATURES if history_length is None else history_length
    if not infosets:
        return torch.empty((0, feature_vector_size(history_length)), device=device, dtype=torch.float32)
    encoded = np.stack([encode_infoset(infoset, history_length=history_length) for infoset in infosets])
    return torch.tensor(encoded, device=device, dtype=torch.float32)

if __name__ == "__main__":
    features = simulate_features()
    buckets, centroids = create_buckets(features)
    np.save('buckets.npy', buckets)
    np.save('centroids.npy', centroids)
    print(f"Buckets created: {len(set(buckets))} unique")