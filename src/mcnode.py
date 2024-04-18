from copy import deepcopy
from enum import Enum
import random
from typing import Optional
import numpy as np


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
        self.is_winner = False

    def __eq__(self, other: object) -> bool:
        return (self.state == other.state) and (self.curr_board == other.curr_board)

    def __str__(self) -> str:
        return f"wins = {self.wins} visits = {self.visits} curr_board = {self.curr_board} is_winner = {self.is_winner} \n children = {self.children}"

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
        self.visits += 1

    def won(self):
        self.wins += 1

    def loss(self):
        self.wins -= 1
    
    def pick_random_child(self):
        c = self.get_random_moves()
        child = self.get_move_in_children(c)
        if child:
            return child
        return self.make_child(c)
    
    def get_move_in_children(self, m):
        for child in self.children:
            if child.curr_board == m:
                return child
        return None

    def make_child(self, move: Optional[int] = None, bd: Optional[int] = None):
        # no point making more children if one is a winner
        if self.is_winner:
            return self.children[0]
        if move == None:
            move = self.get_random_moves()
        child = self.get_move_in_children(move)
        if child and not bd:
            return child
        child = McNode(deepcopy(self.state), bd or self.curr_board, player=self.active_player, parent=self, visits=0)
        child.place_piece(move)

        # if the child is a winner than the parrent is a winner
        if child.check_win():
            self.is_winner = True
            self.children = [child]
        else:
            self.children.append(child)
        return child

    def get_random_moves(self):
        """actually not random, will preference making new paths before going
            down exixting ones
        """
        if self.fully_expanded():
            return random.choice([i for i, x in enumerate(self.state[self.curr_board]) if x == Players.EMPTY.value and i != 0])
        return random.choice([i for i, x in enumerate(self.state[self.curr_board]) if x == Players.EMPTY.value and i != 0 and not self.get_move_in_children(i)])


    def get_fully_expanded(self):
        for i, m in enumerate(self.state[self.curr_board]):
            if m == Players.EMPTY.value and i != 0:
                # note: make child will not make new children if they already exist
                self.make_child(move=i)
        return self.children
    
    def set_parent(self, parent) -> None:
        self.parent = parent

    def fully_expanded(self) -> bool:
        num_children = len(self.children)
        num_blank = np.count_nonzero(self.state[self.curr_board] == Players.EMPTY.value)
        if num_children > num_blank -1:
            raise Exception("too many children")
        return num_children == num_blank -1

    def check_win_board(self, bd) -> bool:
        p = self.get_opposing_player()
        return(  ( bd[1] == p and bd[2] == p and bd[3] == p )
               or( bd[4] == p and bd[5] == p and bd[6] == p )
               or( bd[7] == p and bd[8] == p and bd[9] == p )
               or( bd[1] == p and bd[4] == p and bd[7] == p )
               or( bd[2] == p and bd[5] == p and bd[8] == p )
               or( bd[3] == p and bd[6] == p and bd[9] == p )
               or( bd[1] == p and bd[5] == p and bd[9] == p )
               or( bd[3] == p and bd[5] == p and bd[7] == p ))

    def check_win(self) -> bool:
        if self.parent:
            return self.check_win_board(self.state[self.parent.curr_board])
        else:
            for i in range(1, 10):
                if self.check_win_board(self.state[i]):
                    return True
        return False

    def get_opposing_player(self):
        if self.active_player == Players.PLAYER.value:
            return Players.OPPONENT.value
        return Players.PLAYER.value
