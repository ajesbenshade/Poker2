import random
from collections import defaultdict
import math

ACTIONS = ['p', 'b']  # pass/check or bet/call

class Node:
    def __init__(self, num_actions=2):
        self.num_actions = num_actions
        self.regret_sum = [0.0 for _ in range(num_actions)]
        self.strategy_sum = [0.0 for _ in range(num_actions)]

    def get_strategy(self, reach_prob):
        """Return current strategy via regret matching and update strategy_sum."""
        strategy = [max(r, 0.0) for r in self.regret_sum]
        normalizing = sum(strategy)
        if normalizing > 0:
            strategy = [s / normalizing for s in strategy]
        else:
            strategy = [1.0 / self.num_actions for _ in range(self.num_actions)]
        for i in range(self.num_actions):
            self.strategy_sum[i] += reach_prob * strategy[i]
        return strategy

    def get_average_strategy(self):
        normalizing = sum(self.strategy_sum)
        if normalizing > 0:
            return [s / normalizing for s in self.strategy_sum]
        return [1.0 / self.num_actions for _ in range(self.num_actions)]


def kuhn_terminal(history):
    return history in ('pp', 'bb', 'bp', 'pb')


def kuhn_payoff(history, cards):
    if history == 'pp':
        return 1 if cards[0] > cards[1] else -1
    if history == 'bb':
        return 2 if cards[0] > cards[1] else -2
    if history == 'pb':
        return 1
    if history == 'bp':
        return -1
    raise ValueError(f"Invalid terminal history {history}")


def kuhn_cfr(cards, history, p0, p1, nodes):
    plays = len(history)
    player = plays % 2
    if kuhn_terminal(history):
        return kuhn_payoff(history, cards)
    infoset_key = str(cards[player]) + history
    node = nodes[infoset_key]
    strategy = node.get_strategy(p0 if player == 0 else p1)
    util = [0.0, 0.0]
    node_util = 0.0
    for a, action in enumerate(ACTIONS):
        next_history = history + action
        if player == 0:
            util[a] = -kuhn_cfr(cards, next_history, p0 * strategy[a], p1, nodes)
        else:
            util[a] = -kuhn_cfr(cards, next_history, p0, p1 * strategy[a], nodes)
        node_util += strategy[a] * util[a]
    for a in range(2):
        regret = util[a] - node_util
        if player == 0:
            node.regret_sum[a] += p1 * regret
        else:
            node.regret_sum[a] += p0 * regret
    return node_util


def kuhn_best_response_value(cards, history, player, nodes):
    plays = len(history)
    current_player = plays % 2
    if kuhn_terminal(history):
        payoff = kuhn_payoff(history, cards)
        return payoff if player == 0 else -payoff
    infoset_key = str(cards[current_player]) + history
    node = nodes[infoset_key]
    if current_player == player:
        best = -math.inf
        for action in ACTIONS:
            val = kuhn_best_response_value(cards, history + action, player, nodes)
            if val > best:
                best = val
        return best
    else:
        strat = node.get_average_strategy()
        val = 0.0
        for a, action in enumerate(ACTIONS):
            val += strat[a] * kuhn_best_response_value(cards, history + action, player, nodes)
        return val


def kuhn_exploitability(nodes):
    deck = [1, 2, 3]
    br0 = br1 = 0.0
    deals = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            deals.append((deck[i], deck[j]))
    for cards in deals:
        br0 += kuhn_best_response_value(cards, '', 0, nodes)
        br1 += kuhn_best_response_value(cards, '', 1, nodes)
    br0 /= len(deals)
    br1 /= len(deals)
    return br0 + br1


def test_kuhn_cfr_convergence_and_normalization():
    nodes = defaultdict(Node)
    deck = [1, 2, 3]
    iterations = 20000
    for _ in range(iterations):
        random.shuffle(deck)
        kuhn_cfr(deck[:2], '', 1.0, 1.0, nodes)
    expl = kuhn_exploitability(nodes)
    assert expl < 0.1
    for node in nodes.values():
        strat = node.get_average_strategy()
        assert abs(sum(strat) - 1.0) < 1e-6

# Leduc poker implementation
LEDUC_ACTIONS = ['c', 'b']  # check/call or bet/raise

class LeducNode(Node):
    pass


def leduc_terminal(history, stage):
    if stage == 0:
        return history in ('cc', 'bc', 'cb', 'bb')
    else:
        return len(history) >= 2 and history[-2:] in ('cc', 'bc', 'cb', 'bb')


def leduc_payoff(history, cards, public):
    bets = history.count('b')
    pot = 2 + bets
    if history.endswith('bc'):
        return -1
    if history.endswith('cb'):
        return 1
    rank0 = 3 if cards[0] == public else cards[0]
    rank1 = 3 if cards[1] == public else cards[1]
    return pot / 2 if rank0 > rank1 else -pot / 2


def leduc_cfr(cards, public, history, p0, p1, stage, nodes):
    plays = len(history)
    player = plays % 2
    if leduc_terminal(history, stage):
        return leduc_payoff(history, cards, public)
    infoset_key = f"{cards[player]}{public if public is not None else ''}:{stage}:{history}"
    node = nodes[infoset_key]
    strategy = node.get_strategy(p0 if player == 0 else p1)
    util = [0.0, 0.0]
    node_util = 0.0
    for a, action in enumerate(LEDUC_ACTIONS):
        next_history = history + action
        next_stage = stage
        next_public = public
        if stage == 0 and next_history == 'cc':
            deck = [1, 1, 2, 2, 3, 3]
            deck.remove(cards[0])
            deck.remove(cards[1])
            next_public = deck[0]
            next_stage = 1
            next_history = ''
        if player == 0:
            util[a] = -leduc_cfr(cards, next_public, next_history, p0 * strategy[a], p1, next_stage, nodes)
        else:
            util[a] = -leduc_cfr(cards, next_public, next_history, p0, p1 * strategy[a], next_stage, nodes)
        node_util += strategy[a] * util[a]
    for a in range(2):
        regret = util[a] - node_util
        if player == 0:
            node.regret_sum[a] += p1 * regret
        else:
            node.regret_sum[a] += p0 * regret
    return node_util


def leduc_best_response_value(cards, public, history, player, stage, nodes):
    plays = len(history)
    current_player = plays % 2
    if leduc_terminal(history, stage):
        payoff = leduc_payoff(history, cards, public)
        return payoff if player == 0 else -payoff
    infoset_key = f"{cards[current_player]}{public if public is not None else ''}:{stage}:{history}"
    node = nodes[infoset_key]
    if current_player == player:
        best = -math.inf
        for action in LEDUC_ACTIONS:
            next_history = history + action
            next_stage = stage
            next_public = public
            if stage == 0 and next_history == 'cc':
                deck = [1, 1, 2, 2, 3, 3]
                deck.remove(cards[0])
                deck.remove(cards[1])
                next_public = deck[0]
                next_stage = 1
                next_history = ''
            val = leduc_best_response_value(cards, next_public, next_history, player, next_stage, nodes)
            if val > best:
                best = val
        return best
    else:
        strat = node.get_average_strategy()
        val = 0.0
        for a, action in enumerate(LEDUC_ACTIONS):
            next_history = history + action
            next_stage = stage
            next_public = public
            if stage == 0 and next_history == 'cc':
                deck = [1, 1, 2, 2, 3, 3]
                deck.remove(cards[0])
                deck.remove(cards[1])
                next_public = deck[0]
                next_stage = 1
                next_history = ''
            val += strat[a] * leduc_best_response_value(cards, next_public, next_history, player, next_stage, nodes)
        return val


def leduc_exploitability(nodes):
    deck = [1, 2, 3, 1, 2, 3]
    br0 = br1 = 0.0
    deals = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            deals.append((deck[i], deck[j]))
    for cards in deals:
        br0 += leduc_best_response_value(cards, None, '', 0, 0, nodes)
        br1 += leduc_best_response_value(cards, None, '', 1, 0, nodes)
    br0 /= len(deals)
    br1 /= len(deals)
    return br0 + br1


def test_leduc_cfr_convergence_and_normalization():
    nodes = defaultdict(LeducNode)
    deck = [1, 2, 3, 1, 2, 3]
    iterations = 40000
    for _ in range(iterations):
        random.shuffle(deck)
        cards = deck[:2]
        leduc_cfr(cards, None, '', 1.0, 1.0, 0, nodes)
    expl = leduc_exploitability(nodes)
    assert expl < 0.1
    for node in nodes.values():
        strat = node.get_average_strategy()
        assert abs(sum(strat) - 1.0) < 1e-6
