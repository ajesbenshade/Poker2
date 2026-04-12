import numpy as np

from config import Config
from datatypes import Action, Infoset, Street
from game import simulate_action_batch, terminal


class PokerEnvironment:
    def legal_actions(self, infoset: Infoset):
        raise NotImplementedError

    def next_infoset(self, infoset: Infoset, action: Action):
        raise NotImplementedError

    def evaluate_actions(self, infoset: Infoset, actions):
        raise NotImplementedError

    def evaluate_state(self, infoset: Infoset):
        raise NotImplementedError

    def is_terminal(self, infoset: Infoset, depth=0, max_depth=None):
        raise NotImplementedError


class SimplifiedPokerEnvironment(PokerEnvironment):
    def legal_actions(self, infoset: Infoset):
        return [Action(i) for i in range(Config.NUM_ACTIONS)]

    def next_infoset(self, infoset: Infoset, action: Action):
        next_history = infoset.history + (action.value,)
        next_player = 1 - infoset.acting_player
        next_street = infoset.street
        next_pot_size = infoset.pot_size
        next_stack_sizes = list(infoset.stack_sizes)
        next_current_bet = infoset.current_bet

        if action == Action.FOLD:
            next_current_bet = 0.0
        elif action == Action.CALL:
            call_amount = min(next_current_bet, next_stack_sizes[infoset.acting_player])
            next_stack_sizes[infoset.acting_player] -= call_amount
            next_pot_size += call_amount
        elif action == Action.RAISE:
            raise_amount = min(
                max(next_current_bet, Config.CALL_AMOUNT) * Config.RAISE_MULTIPLIER,
                next_stack_sizes[infoset.acting_player],
            )
            next_stack_sizes[infoset.acting_player] -= raise_amount
            next_pot_size += raise_amount
            next_current_bet = raise_amount

        if len(next_history) % 2 == 0 and next_street != Street.SHOWDOWN:
            next_street = Street(min(next_street.value + 1, Street.SHOWDOWN.value))

        return infoset.next_infoset(
            history=next_history,
            acting_player=next_player,
            street=next_street,
            pot_size=next_pot_size,
            stack_sizes=tuple(next_stack_sizes),
            current_bet=next_current_bet,
        )

    def evaluate_actions(self, infoset: Infoset, actions):
        return simulate_action_batch([infoset] * len(actions), actions)

    def evaluate_state(self, infoset: Infoset):
        actions = self.legal_actions(infoset)
        return float(np.mean(self.evaluate_actions(infoset, actions)))

    def is_terminal(self, infoset: Infoset, depth=0, max_depth=None):
        if max_depth is not None and depth >= max_depth:
            return True
        return terminal(infoset)