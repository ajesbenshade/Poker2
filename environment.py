import logging

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
        acting_stack = max(float(infoset.stack_sizes[infoset.acting_player]), 0.0)
        current_bet = max(float(infoset.current_bet), 0.0)

        actions = [Action.FOLD, Action.CALL]
        min_raise = max(current_bet, Config.CALL_AMOUNT)
        if acting_stack >= min_raise and min_raise > 0.0:
            actions.append(Action.RAISE)
        return actions

    def next_infoset(self, infoset: Infoset, action: Action):
        next_history = infoset.history + (action.value,)
        next_player = 1 - infoset.acting_player
        next_street = infoset.street
        next_pot_size = max(float(infoset.pot_size), 0.0)
        next_stack_sizes = list(infoset.stack_sizes)
        next_current_bet = max(float(infoset.current_bet), 0.0)
        acting_stack = max(next_stack_sizes[infoset.acting_player], 0.0)

        if action == Action.FOLD:
            next_current_bet = 0.0
        elif action == Action.CALL:
            call_amount = min(next_current_bet, acting_stack)
            next_stack_sizes[infoset.acting_player] -= call_amount
            next_pot_size += call_amount
            next_current_bet = 0.0
        elif action == Action.RAISE:
            raise_amount = min(
                max(next_current_bet, Config.CALL_AMOUNT) * Config.RAISE_MULTIPLIER,
                acting_stack,
            )
            if raise_amount <= 0.0:
                logging.warning("Illegal raise with exhausted stack at infoset %s; treating as call.", infoset.key)
                raise_amount = min(next_current_bet, acting_stack)
            next_stack_sizes[infoset.acting_player] -= raise_amount
            next_pot_size += raise_amount
            next_current_bet = raise_amount

        next_stack_sizes = [max(float(stack), 0.0) for stack in next_stack_sizes]
        next_pot_size = max(float(next_pot_size), 0.0)

        if action == Action.FOLD or min(next_stack_sizes) <= 0.0:
            next_street = Street.SHOWDOWN
            next_current_bet = 0.0
        elif len(next_history) >= 2 and next_street != Street.SHOWDOWN:
            last_two_actions = tuple(next_history[-2:])
            betting_round_closed = last_two_actions in {
                (Action.CALL.value, Action.CALL.value),
                (Action.RAISE.value, Action.CALL.value),
            }
            if betting_round_closed:
                next_street = Street(min(next_street.value + 1, Street.SHOWDOWN.value))
                next_current_bet = 0.0

        return infoset.next_infoset(
            history=next_history,
            acting_player=next_player,
            street=next_street,
            pot_size=next_pot_size,
            stack_sizes=tuple(next_stack_sizes),
            current_bet=next_current_bet,
        )

    def evaluate_actions(self, infoset: Infoset, actions):
        return np.nan_to_num(
            simulate_action_batch([infoset] * len(actions), actions),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def evaluate_state(self, infoset: Infoset):
        actions = self.legal_actions(infoset)
        if not actions:
            return 0.0
        value = np.mean(self.evaluate_actions(infoset, actions))
        return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    def is_terminal(self, infoset: Infoset, depth=0, max_depth=None):
        if max_depth is not None and depth >= max_depth:
            return True
        return terminal(infoset)