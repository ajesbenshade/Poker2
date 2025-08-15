# Poker2
Aaron's latest attempt at a GTO poker player

## Abstractions

The project groups similar game states into buckets to make solving manageable.
`abstractions.py` currently builds those buckets with ``k``-means clustering on
simulated equity feature vectors for each board class.  Other schemes such as
effective hand strength (EHS/EHS2) or more elaborate potential-aware methods can
be substituted, but the simple equity-vector approach keeps the example light.

The default bucket counts are:

| Street   | Buckets |
|----------|---------|
| Preflop  | 169     |
| Flop     | 1000    |
| Turn     | 500     |
| River    | 200     |

Hands and boards are canonicalised prior to feature extraction so that suit or
order permutations map to the same bucket, preventing "bucket bleed" where near
equivalent states fall into different clusters.

