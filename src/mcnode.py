from copy import deepcopy
from enum import Enum
import random
from typing import Optional


class Players(Enum):
    EMPTY = 0
    PLAYER = 1
    OPPONENT = 2   
class McNode:
    def __init__(self, state, curr_board, player=Players.PLAYER.value, parent=None, visits=0):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = visits
        self.curr_board = curr_board
        self.active_player = player

    def __eq__(self, other: object) -> bool:
        return (self.state == other.state) and (self.curr_board == other.curr_board)

    def __str__(self) -> str:
        return f"wins = {self.wins} visits = {self.visits} curr_board = {self.curr_board}\n children = {self.children}"

    def swap_player(self):
        if self.active_player == Players.PLAYER.value:
            self.active_player = Players.OPPONENT.value
        else:
            self.active_player = Players.PLAYER.value

    def place_piece(self, move):
        if self.state[self.curr_board][move] != Players.EMPTY.value:
            raise Exception(f"Invalid Move: {move}")
        self.state[self.curr_board][move] = self.active_player
        self.curr_board = move
        self.swap_player()

    def visited(self):
        self.visit += 1

    def won(self):
        self.wins += 1
    
    def pick_random_child(self):
        c = self.get_random_move()
        child = self.get_move_in_children(c)
        if child:
            return child
        return self.make_child(c)
    
    def get_move_in_children(self, m):
        for child in self.children:
            if child.curr_board == m:
                return child
        return None

    def make_child(self, move: Optional[int] = None):
        if move == None:
            move = self.get_random_move()
        child = McNode(deepcopy(self.state), self.curr_board, self.active_player, parent=self, visits=0)
        child.place_piece(move)
        self.children.append(child)
        return child

    def get_random_move(self):
        return random.choice([i for i, x in enumerate(self.state[self.curr_board]) if x == Players.EMPTY.value and i != 0])

    def set_parent(self, parent) -> None:
        self.parent = parent

    def fully_expanded(self) -> bool:
        num_children = len(self.children)
        if num_children > 9:
            raise Exception("too many children")
        return num_children == 9