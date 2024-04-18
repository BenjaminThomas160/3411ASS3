#!/usr/bin/python3
#  agent.py
#  Nine-Board Tic-Tac-Toe Agent starter code
#  COMP3411/9814 Artificial Intelligence
#  CSE, UNSW

import queue
import socket
import sys
from time import sleep
from datetime import datetime
from typing import Optional
import numpy as np
from mcnode import McNode
from copy import deepcopy
import math
import threading
from queue import Queue
import pickle
import os


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

CACHE = './xroot.pkl'
RUNS = 1
BREDTH = 2**12
DEPTH = 20

# the boards are of size 10 because index 0 isn't used
boards = np.zeros((10, 10), dtype="int8")
s = [".","X","O"]
curr = 0 # this is the current board to play in
total_wins = 0


# print a row
def print_board_row(bd, a, b, c, i, j, k):
    print(" "+s[bd[a][i]]+" "+s[bd[a][j]]+" "+s[bd[a][k]]+" | " \
             +s[bd[b][i]]+" "+s[bd[b][j]]+" "+s[bd[b][k]]+" | " \
             +s[bd[c][i]]+" "+s[bd[c][j]]+" "+s[bd[c][k]])

# Print the entire board
def print_board(board):
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
    return(  ( bd[1] == p and bd[2] == p and bd[3] == p )
           or( bd[4] == p and bd[5] == p and bd[6] == p )
           or( bd[7] == p and bd[8] == p and bd[9] == p )
           or( bd[1] == p and bd[4] == p and bd[7] == p )
           or( bd[2] == p and bd[5] == p and bd[8] == p )
           or( bd[3] == p and bd[6] == p and bd[9] == p )
           or( bd[1] == p and bd[5] == p and bd[9] == p )
           or( bd[3] == p and bd[5] == p and bd[7] == p ))

def board_nearly_won(p, bd) -> int:
    o = swap_player(p)
    return int(
        max((bd[1] == p) + (bd[2] == p) + (bd[3] == p) - 2*((bd[1] == o) + (bd[2] == o) + (bd[3] == o)), 0) +
        max((bd[4] == p) + (bd[5] == p) + (bd[6] == p) - 2*((bd[4] == o) + (bd[5] == o) + (bd[6] == o)), 0) +
        max((bd[7] == p) + (bd[8] == p) + (bd[9] == p) - 2*((bd[7] == o) + (bd[8] == o) + (bd[9] == o)), 0) +
        max((bd[1] == p) + (bd[4] == p) + (bd[7] == p) - 2*((bd[1] == o) + (bd[4] == o) + (bd[7] == o)), 0) +
        max((bd[2] == p) + (bd[5] == p) + (bd[8] == p) - 2*((bd[2] == o) + (bd[5] == o) + (bd[8] == o)), 0) +
        max((bd[3] == p) + (bd[6] == p) + (bd[9] == p) - 2*((bd[3] == o) + (bd[6] == o) + (bd[9] == o)), 0) +
        max((bd[1] == p) + (bd[5] == p) + (bd[9] == p) - 2*((bd[3] == o) + (bd[6] == o) + (bd[9] == o)), 0) +
        max((bd[3] == p) + (bd[5] == p) + (bd[7] == p) - 2*((bd[3] == o) + (bd[5] == o) + (bd[7] == o)), 0)
    )

def game_won( player: int, boardz: np.array ) -> bool:
    for b in range(1,len(boardz)):
        if board_won(player, boardz[b]):
            return True
    return False

def game_over( curr_board: np.array, boardz: np.array ) -> bool:
    if game_won(PLAYER, boardz) or game_won(OPPONENT, boardz):
        return True
    if np.count_nonzero(boardz[curr_board] != EMPTY) == 9:
        return False
    return False

def swap_player(p):
    if p == PLAYER:
        return OPPONENT
    return PLAYER

def heuristic(player: int, boardz: np.array) -> int:
    out = 0
    for i in range(1,10):
        board_heuristic = np.count_nonzero(boardz[i] == player) - np.count_nonzero(boardz[i] == swap_player(player))
        out += board_heuristic**5
    return out
        
def sim_rand_game(node: McNode , m: int) -> int:
    if m >= DEPTH:
        return 0
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
    q: queue.Queue,
    root: Optional[McNode] = None
) -> McNode:
    # start_time = datetime.now()
    if not root:
        root = McNode(deepcopy(boardz), curr_board, player=player)
    # while (datetime.now() - start_time).total_seconds() < 2.5:
    for _ in range(BREDTH):
        node = root
        while node.fully_expanded():
            node = max(node.children, key=lambda x: x.wins / x.visits + math.sqrt(2 * math.log(node.visits) / x.visits))
        
        winner = sim_rand_game(node, 0)

        while node is not root.parent:
            node.visited()
            if winner == node.get_opposing_player():
                node.won()
            elif winner == node.active_player:
                node.loss()
            node = node.parent
    q.put_nowait(root)
    return root

def begin_threading(
    player: int,
    boardz: np.array,
    curr_board: int,
    q: queue.Queue,
    root: Optional[McNode]
):
    if not root:
        root = McNode(deepcopy(boardz), curr_board, player=player)
    print(root)
    children = root.get_fully_expanded()
    print(children)
    if not children:
        return montecarl(player, boardz, curr_board, q, root)
    threads = []
    for child in children:
        child.set_parent(None)
        threads.append(threading.Thread(target=montecarl, args=(child.active_player, child.state, child.curr_board, q, child)))
    print(threads) 
    for i in threads:
        i.start()
 
    print("thingo: ")
    for c in threads:
        c.join()
    res = get_most_wins(q)
    return res

def get_most_wins(q):
    if q.empty():
        return -1
    maxx = float('-inf')
    while not q.empty():
        temp = q.get()
        maxx = max(maxx, temp.wins)
 
    return maxx

def get_num_moves(b):
    m = 0
    for i in range(0,9):
        if b[i] != EMPTY:
            m += 1
    return m

def make_move( player, m, move, board ):
    if board[move[m]] != EMPTY:
        print('Illegal Move')
        return ILLEGAL_MOVE
    else:
        board[move[m]] = player
        if game_won( player, board ):
            return WIN
        elif full_board( board ):
            return DRAW
        else:
            return STILL_PLAYING

def full_board( board ):
    b = 1
    while b <= 9 and board[b] != EMPTY:
        b += 1
    return( b == 10 )



#best_move = np.zeros(81,dtype=np.int32)
curr_best_child: Optional[McNode] = None
move = 0
# choose a move to play
def play(m: int, r: Optional[McNode]):
    global curr_best_child
    root = montecarl(PLAYER, deepcopy(boards), curr, r)

    best_child = max(root.children, key=lambda x: x.visits)
    curr_best_child = best_child

    best_move = best_child.curr_board
    place(curr, best_move, PLAYER)
    print(f"bestmove {m}: {best_move}, visited: {best_child.visits}, wins: {best_child.wins}")

    return best_move
    
# place a move in the global boards
def place( board, num, player ):
    global curr
    curr = num
    boards[board][num] = player

# read what the server sent us and
# parse only the strings that are necessary
def parse(r: McNode, string: str):
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
        curr_best_child = r.make_child(bd=int(args[0]), move=int(args[1]))
        move = 1
        return play(1, curr_best_child)  # choose and return the second move

    # third_move(K,L,M) means that the first and second move were
    # in square L of sub-board K, and square M of sub-board L,
    # and we are expected to return the third move.
    elif command == "third_move":
        # place the first move (randomly generated for us)
        place(int(args[0]), int(args[1]), 1)
        curr_best_child = r.make_child(bd=int(args[0]), move=int(args[1]))
        # place the second move (chosen by opponent)
        place(curr, int(args[2]), 2)
        root = curr_best_child.make_child(move=int(args[2]))
        move = 2
        return play(2, root) # choohe and return the third move

    # nex_move(M) means that the previous move was into
    # square M of the designated sub-board,
    # and we are expected to return the next move.
    elif command == "next_move":
        # place the previous move (chosen by opponent)
        place(curr, int(args[0]), 2)
        root = curr_best_child.make_child(move=int(args[0]))
        move += 2
        return play(move, root) # choose and return our next move

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

    if os.path.isfile(CACHE):
        with open(CACHE, "rb") as f:
            super_root = pickle.loads(f.read())
    else:
        super_root = McNode(deepcopy(boards), PLAYER)
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
            response = parse(super_root, line)
            if response == -1:
                sleep(1)
                # s.close()
                # return
                boards = np.zeros((10, 10), dtype="int8")
                curr_best_child = None
                curr = 0
                i += 1
                print(f"game: {i}, wins: {total_wins}")
                if i == RUNS:
                    # print("Saving")
                    # pkl = pickle.dumps(super_root)
                    # with open(CACHE, "wb+") as f:
                    #    f.write(pkl)
                    return
            elif response > 0:
                s.sendall((str(response) + "\n").encode())

if __name__ == "__main__":
    main()
