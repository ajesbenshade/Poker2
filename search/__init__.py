"""Real-time search for Poker2: GPU subgame solvers built on PyTorch.

Card ids match the C++ engine: id = rank * 4 + suit, rank 0..12 = 2..A,
suit 0..3 = s,h,d,c. Hole-card combos are numbered 0..1325 in the same order
as the engine (a < b, lexicographic).
"""
