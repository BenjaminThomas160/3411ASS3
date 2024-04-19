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

ILLEGAL_MOVE  = 0
STILL_PLAYING = 1
WIN           = 2
LOSS          = 3
DRAW          = 4

MAX_MOVE      = 9

MIN_EVAL = -1000000
MAX_EVAL =  1000000

RUNS = 1
BREDTH = 2**10
DEPTH = 10
MAX_WINS = 2**11
TIME_SIMULATING = 2.2

# the boards are of size 10 because index 0 isn't used
boards = np.zeros((10, 10), dtype="int8")
s = [".","X","O"]
curr = 0 # this is the current board to play in
total_wins = 0


# print a row
def print_board_row(bd, a, b, c, i, j, k):
    """Prints a row of the board

    Args:
        bd (np.array): the entire board
        a (int): A location in the row 
        b (int): A location in the row
        c (int): A location in the row
        i (int): A location in the row
        j (int): A location in the row
        k (int): A location in the row
    """
    print(" "+s[bd[a][i]]+" "+s[bd[a][j]]+" "+s[bd[a][k]]+" | " \
             +s[bd[b][i]]+" "+s[bd[b][j]]+" "+s[bd[b][k]]+" | " \
             +s[bd[c][i]]+" "+s[bd[c][j]]+" "+s[bd[c][k]])

# Print the entire board
def print_board(board):
    """Prints the entire board

    Args:
        board (np.array): the board
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


def board_won( p, bd ):
    """returns bool if a single board won or not

    Args:
        p (int): The player
        bd (np.array): The board
    """
    return(  ( bd[1] == p and bd[2] == p and bd[3] == p )
           or( bd[4] == p and bd[5] == p and bd[6] == p )
           or( bd[7] == p and bd[8] == p and bd[9] == p )
           or( bd[1] == p and bd[4] == p and bd[7] == p )
           or( bd[2] == p and bd[5] == p and bd[8] == p )
           or( bd[3] == p and bd[6] == p and bd[9] == p )
           or( bd[1] == p and bd[5] == p and bd[9] == p )
           or( bd[3] == p and bd[5] == p and bd[7] == p ))

def game_won( player: int, boardz: np.array ) -> bool:
    """Returns whether the game is won

    Args:
        player (int): The active player
        boardz (np.array): The entire board

    Returns:
        bool: If the game is won or not
    """
    for b in range(1,len(boardz)):
        if board_won(player, boardz[b]):
            return True
    return False

def game_over( curr_board: np.array, boardz: np.array ) -> bool:
    """Determines if the game is complete

    Args:
        curr_board (np.array): The current board
        boardz (np.array): The entire board

    Returns:
        bool: whether the game is over or not
    """
    if game_won(PLAYER, boardz) or game_won(OPPONENT, boardz):
        return True
    if np.count_nonzero(boardz[curr_board] != EMPTY) == 9:
        return False
    return False

def swap_player(p):
    """Swaps the player

    Args:
        p (int): The current player

    Returns:
        Int: The other player 
    """
    if p == PLAYER:
        return OPPONENT
    return PLAYER

def sim_rand_game(node: McNode , m: int) -> int:
    """Simulates a random game

    Args:
        node (McNode): The current node in the tree
        m (int): the current move

    Returns:
        int: The player who won, or a DRAW
    """
    if m >= DEPTH:
        return DRAW
    if node.is_winner:
        return node.active_player

    board = node.state[node.curr_board]
    if np.count_nonzero(board != EMPTY) == 9:
        return 0

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
    boardz: np.array,
    curr_board: int,
    root: Optional[McNode] = None
) -> McNode:
    """Does the MonteCarlo Tree Search algorithm

    Args:
        player (int): The current player
        boardz (np.array): The entire board
        curr_board (int): The current subboard
        root (Optional[McNode], optional): The current root node. Defaults to None.

    Returns:
        McNode: The node with the most visits
    """
    start_time = datetime.now()
    if not root:
        root = McNode(deepcopy(boardz), curr_board, player=player)
    while (datetime.now() - start_time).total_seconds() < TIME_SIMULATING:
        node = root
        if abs(node.wins) >= MAX_WINS:
            return root
        while node.fully_expanded():
            node = max(
                node.children, key=lambda x: x.wins / x.visits
                + math.sqrt(2 * math.log(node.visits) / x.visits)
            )

        winner = sim_rand_game(node, 0)

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
    return montecarl(root.active_player, root.state, root.curr_board)

def begin_multiprocessing(
    player: int,
    boardz: np.array,
    curr_board: int,
    p: mp.Pool,
    root: Optional[McNode]
) -> McNode:
    """Does the multiprocessing logic

    Args:
        player (int): The current player
        boardz (np.array): The entire board
        curr_board (int): The current subboard
        p (mp.Pool): The processing pool
        root (Optional[McNode]): The first root node

    Raises:
        Exception: no children

    Returns:
        McNode: The node with the highest win %
    """
    if not root:
        root = McNode(deepcopy(boardz), curr_board, player=player)
    children = root.get_fully_expanded()
    if not children:
        raise Exception("No children")
    results = p.map(montecarl_wrapper, root.children)
    res = max(results, key=lambda x: x.wins / x.visits)
    return res

def get_most_win_percentage(q):
    if q.empty():
        return -1
    max_node = q.get()
    max_val = max_node.wins / max_node.visits
    while not q.empty():
        temp = q.get()
        new_val = temp.wins / temp.visits
        if max_val < new_val:
            max_node = temp
            max_val = new_val
    return max_node

def get_num_moves(b):
    m = 0
    for i in range(0,9):
        if b[i] != EMPTY:
            m += 1
    return m

curr_best_child: Optional[McNode] = None

def play(p: mp.Pool, r: Optional[McNode] = None):
    """Choose a move to play

    Args:
        p (mp.Pool): The processing pool
        r (Optional[McNode], optional): Optional root node. Defaults to None.

    Returns:
        Int: The best move
    """
    global curr_best_child
    best_child = begin_multiprocessing(PLAYER, deepcopy(boards), curr, p, r)
    curr_best_child = best_child
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
    global move
    global curr_best_child
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
        # root = curr_best_child.make_child(move=int(args[2]))
        return play(p) # choohe and return the third move

    # nex_move(M) means that the previous move was into
    # square M of the designated sub-board,
    # and we are expected to return the next move.
    elif command == "next_move":
        # place the previous move (chosen by opponent)
        place(curr, int(args[0]), 2)
        root = curr_best_child.make_child(move=int(args[0]))
        return play(p, root) # choose and return our next move

    elif command == "win":
        print("Yay!! We win!! :)")
        global total_wins
        total_wins += 1
        return -1

    elif command == "loss":
        print("We lost :(")
        return -1

    return 0

# connect to socket
def main():
    global boards
    global curr_best_child
    global curr

    with mp.Pool(processes=9) as p:

        print("ready")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = int(sys.argv[2]) # Usage: ./agent.py -p (port)
        s.connect(('localhost', port))
        i = 0

        while True:
            text = s.recv(1024).decode()
            if not text:
                continue
            for line in text.split("\n"):
                start_time = datetime.now()
                response = parse(p, line)
                if response == -1:
                    boards = np.zeros((10, 10), dtype="int8")
                    curr_best_child = None
                    curr = 0
                    i += 1
                    if i == RUNS:
                        return
                elif response > 0:
                    s.sendall((str(response) + "\n").encode())

if __name__ == "__main__":
    main()
