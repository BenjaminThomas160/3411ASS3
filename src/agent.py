#!/usr/bin/python3
#  agent.py
#  Nine-Board Tic-Tac-Toe Agent starter code
#  COMP3411/9814 Artificial Intelligence
#  CSE, UNSW

import socket
import sys
from copy import deepcopy
from datetime import datetime
import math
import multiprocessing as mp
from typing import Optional
import numpy as np
from mcnode import McNode


# a board cell can hold:
#   0 - Empty
#   1 - We played here
#   2 - Opponent played here

EMPTY = 0
PLAYER = 1
OPPONENT = 2

WIN           = 2
LOSS          = 3
DRAW          = 4
SUB_BOARD_SIZE = 9

RUNS = 1
BREDTH = 2**10
DEPTH = 10
MAX_WINS = 2**11
TIME_SIMULATING = 2.2

# the boards are of size 10 because index 0 isn't used
boards = np.zeros((10, 10), dtype="int8")
curr = 0 # this is the current board to play in


# print a row
def print_board_row(
    bd: np.ndarray,
    a: int,
    b: int,
    c: int,
    i: int,
    j: int,
    k: int,
) -> None:
    """Prints a row of the board

    Args:
        bd (np.ndarray): the entire board
        a (int): A location in the row 
        b (int): A location in the row
        c (int): A location in the row
        i (int): A location in the row
        j (int): A location in the row
        k (int): A location in the row
    """
    s = [".","X","O"]
    print(" "+s[bd[a][i]]+" "+s[bd[a][j]]+" "+s[bd[a][k]]+" | " \
             +s[bd[b][i]]+" "+s[bd[b][j]]+" "+s[bd[b][k]]+" | " \
             +s[bd[c][i]]+" "+s[bd[c][j]]+" "+s[bd[c][k]])

# Print the entire board
def print_board(board: np.ndarray) -> None:
    """Prints the entire board

    Args:
        board (np.ndarray): the board
    """
    print_board_row(board, 1,2,3,1,2,3)
    print_board_row(board, 1,2,3,4,5,6)
    print_board_row(board, 1,2,3,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 4,5,6,1,2,3)
    print_board_row(board, 4,5,6,4,5,6)
    print_board_row(board, 4,5,6,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 7,8,9,1,2,3)
    print_board_row(board, 7,8,9,4,5,6)
    print_board_row(board, 7,8,9,7,8,9)
    print()

def sim_rand_game(node: McNode , m: int) -> int:
    """Simulates a random game

    Args:
        node (McNode): The current node in the tree
        m (int): the current move

    Returns:
        int: The player who won, or a DRAW
    """
    if m >= DEPTH:
        # if we have reached max depth assume it is a draw
        return DRAW
    # heuristic value to guide decision
    if node.is_winner:
        return node.active_player

    board = node.state[node.curr_board]
    # if draw
    if np.count_nonzero(board != EMPTY) == SUB_BOARD_SIZE:
        return DRAW

    if node.check_win():
        return node.get_opposing_player()

    new_node = node.pick_random_child()

    res = sim_rand_game(new_node, m+1)
    new_node.visited()
    if res == new_node.get_opposing_player():
        new_node.won()
    elif res == new_node.active_player:
        new_node.loss()
    return res

def montecarl(
    player: int,
    boardz: np.ndarray,
    curr_board: int,
    root: Optional[McNode] = None
) -> McNode:
    """Does the MonteCarlo Tree Search algorithm

    Args:
        player (int): The current player
        boardz (np.ndarray): The entire board
        curr_board (int): The current subboard
        root (Optional[McNode], optional): The current root node. Defaults to None.

    Returns:
        McNode: The node with the most visits
    """
    start_time = datetime.now()
    if not root:
        root = McNode(
            state = deepcopy(boardz),
            curr_board=curr_board,
            player=player,
        )
    while (datetime.now() - start_time).total_seconds() < TIME_SIMULATING:
        node = root
        # speed up by returning obvious wins and losses
        if abs(node.wins) >= MAX_WINS:
            return root
        # Upper confindence bound trees
        while node.fully_expanded():
            node = max(
                node.children, key=lambda x: x.wins / x.visits
                + math.sqrt(2 * math.log(node.visits) / x.visits)
            )
        # Simulare a game and get the winner
        winner = sim_rand_game(node, 0)
        # backpropegate up the tree to the root and update wins and losses
        while node is not root.parent:
            node.visited()
            if winner == node.get_opposing_player():
                node.won()
            elif winner == node.active_player:
                node.loss()
            node = node.parent
    return root

def montecarl_wrapper(root):
    """wrapper around montecarl for multiprocessing

    Args:
        root (McNode): The root node to hand to montecarl

    Returns:
        McNode: The node with the most visits
    """
    return montecarl(root.active_player, root.state, root.curr_board, root)

def begin_multiprocessing(
    player: int,
    boardz: np.ndarray,
    curr_board: int,
    p: mp.Pool,
) -> McNode:
    """Does the multiprocessing logic

    Args:
        player (int): The current player
        boardz (np.ndarray): The entire board
        curr_board (int): The current subboard
        p (mp.Pool): The processing pool
        root (Optional[McNode]): The first root node

    Returns:
        McNode: The node with the highest win %
    """
    root = McNode(deepcopy(boardz), curr_board, player=player)
    children = root.get_fully_expanded()
    # Orphan children
    for child in children:
        child.set_parent(None)

    results = p.map(montecarl_wrapper, children)
    res = max(results, key=lambda x: x.wins / x.visits)
    return res

def play(p: mp.Pool):
    """Choose a move to play

    Args:
        p (mp.Pool): The processing pool

    Returns:
        Int: The best move
    """
    best_child = begin_multiprocessing(PLAYER, boards, curr, p)
    best_move = best_child.curr_board
    place(curr, best_move, PLAYER)
    return best_move

# place a move in the global boards
def place( board, num, player ):
    global curr
    curr = num
    boards[board][num] = player

# read what the server sent us and
# parse only the strings that are necessary
def parse(p: mp.Pool, string: str):
    if "(" in string:
        command, args = string.split("(")
        args = args.split(")")[0]
        args = args.split(",")
    else:
        command, args = string, []

    # init tells us that a new game is about to begin.
    # start(x) or start(o) tell us whether we will be playing first (x)
    # or second (o); we might be able to ignore start if we internally
    # use 'X' for *our* moves and 'O' for *opponent* moves.

    # second_move(K,L) means that the (randomly generated)
    # first move was into square L of sub-board K,
    # and we are expected to return the second move.
    if command == "second_move":
        # place the first move (randomly generated for opponent)
        place(int(args[0]), int(args[1]), 2)
        return play(p)  # choose and return the second move

    # third_move(K,L,M) means that the first and second move were
    # in square L of sub-board K, and square M of sub-board L,
    # and we are expected to return the third move.
    elif command == "third_move":
        # place the first move (randomly generated for us)
        place(int(args[0]), int(args[1]), 1)
        # place the second move (chosen by opponent)
        place(curr, int(args[2]), 2)
        return play(p) # choohe and return the third move

    # nex_move(M) means that the previous move was into
    # square M of the designated sub-board,
    # and we are expected to return the next move.
    elif command == "next_move":
        # place the previous move (chosen by opponent)
        place(curr, int(args[0]), 2)
        return play(p) # choose and return our next move

    elif command == "win":
        print("Yay!! We win!! :)")
        return -1

    elif command == "loss":
        print("We lost :(")
        return -1

    return 0

# connect to socket
def main():
    with mp.Pool(processes=9) as p:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = int(sys.argv[2]) # Usage: ./agent.py -p (port)
        s.connect(('localhost', port))
        while True:
            text = s.recv(1024).decode()
            if not text:
                continue
            for line in text.split("\n"):
                response = parse(p, line)
                if response == -1:
                    s.close()
                    return
                elif response > 0:
                    s.sendall((str(response) + "\n").encode())

if __name__ == "__main__":
    main()
