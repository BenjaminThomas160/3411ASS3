from copy import deepcopy
from enum import Enum
import random
import numpy as np


class Players(Enum):
    """Enum representing different types of players in a game.

    Attributes:
        EMPTY (int): Represents an empty space on the game board.
        PLAYER (int): Represents the player.
        OPPONENT (int): Represents the opponent.
    """
    EMPTY = 0
    PLAYER = 1
    OPPONENT = 2

class McNode:
    """Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.

    Args:
        state (numpy.ndarray): The state of the game.
        curr_board (int): The current board the game is in.
        player (int, optional): The active player. Defaults to Players.PLAYER.value.
        parent (McNode, optional): The parent node. Defaults to None.

    Attributes:
        state (numpy.ndarray): The state of the game.
        parent (McNode): The parent node.
        children (list): List of child McNodes.
        wins (int): Number of wins associated with this node.
        visits (int): Number of times this node has been visited.
        curr_board (int): The current sub board the game is in .
        active_player (int): The active player.
        is_winner (bool): Indicates if this node represents a winning state.

    Raises:
        ValueError: If an invalid move is attempted.

    """
    def __init__(
        self,
        state: np.ndarray,
        curr_board: int,
        player: int = Players.PLAYER.value,
        parent = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.curr_board = curr_board
        self.active_player = player
        self.is_winner = False

    def __eq__(self, other) -> bool:
        return (self.state == other.state) and (self.curr_board == other.curr_board)

    def swap_player(self) -> None:
        """swaps the active player"""
        if self.active_player == Players.PLAYER.value:
            self.active_player = Players.OPPONENT.value
        else:
            self.active_player = Players.PLAYER.value

    def place_piece(self, move: int) -> None:
        """Places a piece at move on its state

        Args:
            move (int): The move to make
        """
        self.state[self.curr_board][move] = self.active_player
        self.curr_board = move
        self.swap_player()

    def visited(self) -> None:
        """visits this node"""
        self.visits += 1

    def won(self) -> None:
        """Adds win to this node"""
        self.wins += 1

    def loss(self) -> None:
        """Adds loss to the node"""
        self.wins -= 1

    def pick_random_child(self):
        """picks a random child and makes it if it doesn't exit, for montecarlo tree search"""
        move, exists = self.get_random_moves()
        if exists:
            return self.get_move_in_children(move)
        return self.make_child(move)

    def get_move_in_children(self, m: int):
        """If the move is already made in the children node, then return it

        Args:
            m (int): The move

        Returns:
            McNode: The child node with the move, or None if no child node has that move
        """
        for child in self.children:
            if child.curr_board == m:
                return child
        return None

    def make_child(
        self,
        move: int,
    ):
        """Makes a new child for a given move and adds it to this node

        Args:
            move (int): The move for the child.

        Returns:
            McNode: The new Child
        """
        # no point making more children if one is a winner
        if self.is_winner:
            return self.children[0]
        child = McNode(
            deepcopy(self.state),
            self.curr_board,
            player=self.active_player,
            parent=self,
        )
        child.place_piece(move)

        # if the child is a winner than the parrent is a winner
        if child.check_win():
            self.is_winner = True
            self.children = [child]
        else:
            self.children.append(child)
        return child

    def get_random_moves(self) -> tuple[int, bool]:
        """Gets random move, actually not random, will preference making new paths before going
            down existing ones
            returns:
                tuple[int, bool]: an int representing the move
                    and a bool representing if that node exists
        """
        if self.fully_expanded():
            return (random.choice(
                [i for i, x in enumerate(self.state[self.curr_board])
                 if x == Players.EMPTY.value and i != 0]
                ), True)

        return (random.choice(
            [i for i, x in enumerate(self.state[self.curr_board])
             if x == Players.EMPTY.value and i != 0
             and not self.get_move_in_children(i)]
            ), False)


    def get_fully_expanded(self) -> list:
        """fully expands the child nodes

        Returns:
            List(McNode): List of the children, now fully expanded
        """
        if self.fully_expanded():
            return self.children

        for i, m in enumerate(self.state[self.curr_board]):
            if m == Players.EMPTY.value and i != 0 and not self.get_move_in_children(i):
                self.make_child(move=i)
        return self.children

    def set_parent(self, parent) -> None:
        """Sets the parent

        Args:
            parent (McNode): parent to be set
        """
        self.parent = parent

    def fully_expanded(self) -> bool:
        """Returns whether or not the current node is fully expanded

        Returns:
            bool: True if fully expanded, False otherwise
        """
        num_children = len(self.children)
        num_blank = np.count_nonzero(self.state[self.curr_board] == Players.EMPTY.value)
        return num_children == num_blank -1

    def check_win_board(self, bd: np.ndarray) -> bool:
        """Check if the board is won by the opposing player

        Args:
            bd (np.ndarray): The board to be checked

        Returns:
            bool: If won or not
        """
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
        """Checks if the entire self.state board is won

        Returns:
            bool: True if won, false if not
        """
        # if we have a parrent we know which board the win will come from
        if self.parent:
            return self.check_win_board(self.state[self.parent.curr_board])
        for i in range(1, 10):
            if self.check_win_board(self.state[i]):
                return True
        return False

    def get_opposing_player(self):
        """Gets the opposing player

        Returns:
            Int: The opposing player
        """
        if self.active_player == Players.PLAYER.value:
            return Players.OPPONENT.value
        return Players.PLAYER.value
