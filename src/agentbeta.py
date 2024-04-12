#!/usr/bin/python3
#  agent.py
#  Nine-Board Tic-Tac-Toe Agent starter code
#  COMP3411/9814 Artificial Intelligence
#  CSE, UNSW

import socket
import sys
from typing import Optional
import numpy as np
from mcnode import McNode
from copy import deepcopy
import random
import math

# a board cell can hold:
#   0 - Empty
#   1 - We played here
#   2 - Opponent played here

EMPTY = 0
PLAYER = 2
OPPONENT = 1

ILLEGAL_MOVE  = 0
STILL_PLAYING = 1
WIN           = 2
LOSS          = 3
DRAW          = 4

MAX_MOVE      = 9

MIN_EVAL = -1000000
MAX_EVAL =  1000000

# the boards are of size 10 because index 0 isn't used
boards = np.zeros((10, 10), dtype="int8")
s = [".","X","O"]
curr = 0 # this is the current board to play in

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

def map_to_board(a,b):
    s = 0
    if a == 1:
        s == 4
    elif a == 2:
        s == 7
    return s + b

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

def game_won( player, boards ):
    for b in range(1,len(boards)):
        if board_won(player, boards[b]):
            return True
    return False

def swap_player(p):
    if p == PLAYER:
        return OPPONENT
    return PLAYER

abcache = {}

def ab_to_hash(
    player: int,
    boards: np.array,
    alpha: int,
    beta: int,
    curr_board: int
) -> str:
    return f"{player},{boards.tobytes()},{alpha},{beta},{curr_board}"

def heuristic(player: int, boards: np.array) -> int:
    out = 0
    for i in range(1,10):
        board_heuristic = np.count_nonzero(boards[i] == player) - np.count_nonzero(boards[i] == swap_player(player))
        out += board_heuristic**3
    return out
    """
    out = 0
    for i in range(1,10):
        board_heuristic = board_nearly_won(player, boards[i]) - board_nearly_won(swap_player(player), boards[i])
        out += board_heuristic**5
    return out
    """
        


def alphabeta(
    player: int,
    m: int,
    boards: np.array,
    alpha: int,
    beta: int,
    best_move: tuple[int,int],
    curr_board: int
) -> int:
    #if ab_to_hash(player, boards, alpha, beta, curr_board) in abcache:
    #    return abcache[ab_to_hash(player, boards, alpha, beta, curr_board)]
    
    board = boards[curr_board] 
    best_eval = MIN_EVAL

    if game_won( swap_player(player), boards ):   # LOSS
        return -1000 + m  # better to win faster (or lose slower)
    
    if m - move == 7:
        h = heuristic(player, boards)
        return h - m

    this_move = 0
    for r in range(1, 10):
        if board[r] == EMPTY:         # move is legal

            this_move = r
            board[this_move] = player # make move
            this_eval = -alphabeta(swap_player(player), m+1, deepcopy(boards), -beta, -alpha, best_move, this_move)
    #        abcache[ab_to_hash(swap_player(player), boards, -beta, -alpha, this_move)] = -this_eval
            board[this_move] = EMPTY  # undo move
            if this_eval > best_eval:
                best_move[m] = this_move
                best_eval = this_eval
                if best_eval > alpha:
                    alpha = best_eval
                    if alpha >= beta: # cutoff
                        return( alpha )

    if this_move == 0:  # no legal moves
        return( 0 )     # DRAW
    else:
        return( alpha )

def sim_rand_game(player, node):
    board = node.state[node.curr_board]
    if np.count_nonzero(board != EMPTY) == 9:
        return 0
    child_board = node.state.copy()
    random_move = random.choice([i for i, x in enumerate(child_board[node.curr_board]) if x == EMPTY and i != 0])
    child_board[node.curr_board][random_move] = player
    new_node = McNode(child_board, random_move, parent=node)
    check_existing = list(filter(lambda x: x == new_node, node.children))
    if check_existing:
        new_node = check_existing[0]
    else:
        node.children.append(new_node) 
    if game_won(player, new_node.state):
        return player
    else:
        sim_rand_game(swap_player(player), new_node)



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



best_move = np.zeros(81,dtype=np.int32)
move = 0
# choose a move to play
def play(m):
#    m = get_num_moves(boards[curr])
    alphabeta(PLAYER, m, boards, MIN_EVAL, MAX_EVAL, best_move, curr)
    #root = montecarl(PLAYER, boards, curr)
    #best_child = max(root.children, key=lambda x: x.visits)
    #best_move = best_child.curr_board
   # root.state.index([x for x in best_child.state if x != boards[curr]][0])
    place(curr, best_move[m], PLAYER)
 #   print(f"board: {curr} move: {best_move} {m}")
    print(f"bestmove : {best_move}")

    return best_move[m]
    
# place a move in the global boards
def place( board, num, player ):
    global curr
    curr = num
    boards[board][num] = player

# read what the server sent us and
# parse only the strings that are necessary
def parse(string):
    global move
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
        move = 1
        return play(1)  # choose and return the second move

    # third_move(K,L,M) means that the first and second move were
    # in square L of sub-board K, and square M of sub-board L,
    # and we are expected to return the third move.
    elif command == "third_move":
        # place the first move (randomly generated for us)
        place(int(args[0]), int(args[1]), 1)
        # place the second move (chosen by opponent)
        place(curr, int(args[2]), 2)
        move = 2
        return play(2) # choose and return the third move

    # nex_move(M) means that the previous move was into
    # square M of the designated sub-board,
    # and we are expected to return the next move.
    elif command == "next_move":
        # place the previous move (chosen by opponent)
        place(curr, int(args[0]), 2)
        move += 2
        return play(move) # choose and return our next move

    elif command == "win":
        print("Yay!! We win!! :)")
        return -1

    elif command == "loss":
        print("We lost :(")
        return -1

    return 0

# connect to socket
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(sys.argv[2]) # Usage: ./agent.py -p (port)

    s.connect(('localhost', port))
    while True:
        text = s.recv(1024).decode()
        if not text:
            continue
        for line in text.split("\n"):
            response = parse(line)
            if response == -1:
                s.close()
                return
            elif response > 0:
                s.sendall((str(response) + "\n").encode())

if __name__ == "__main__":
    main()
