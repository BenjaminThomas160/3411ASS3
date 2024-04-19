#**********************************************************
#   ttt.py
#
#   UNSW CSE
#   COMP3411/9814
#   Code for Tic-Tac-Toe with Alpha-Beta search
#
import numpy as np
import torch
from torch.nn import MSELoss
from nn import NeuralNetwork, NetContext

EMPTY = 0

ILLEGAL_MOVE  = 5
STILL_PLAYING = 1
WIN           = 2
LOSS          = 3
DRAW          = 4

MAX_MOVE      = 9

MIN_EVAL = -1000000
MAX_EVAL =  1000000

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def get_net(player):
    policy_net = NeuralNetwork().to(device)
    target_net = NeuralNetwork().to(device)
    sgd = torch.optim.SGD(policy_net.parameters(), lr=0.1)
    loss = MSELoss()
    net_context = NetContext(player, policy_net, target_net, sgd, loss)
    return net_context

def swap_player(p):
    if p == 1:
        return 2
    return 1

def game(p1, p2) -> tuple[int, list[tuple]]:
    board = EMPTY*np.ones(10,dtype=np.int32)
    move = np.zeros(10,dtype=np.int32)
    move_history = []
    game_status = STILL_PLAYING
    player = 1
    m = 0
    while m < MAX_MOVE and game_status == STILL_PLAYING:
        m += 1
        player = swap_player(player)
        inp = torch.from_numpy(board[1:]/2).float() 
        inp = inp.unsqueeze(0)

        if player == 1:
            fd = p1(inp).detach().numpy()
        else:
            fd = p2(inp).detach().numpy()

        move[m] = np.argmax(fd)
        if move[m] < 1 or move[m] > 9 or board[move[m]] != EMPTY:
            print_board( board )
            return (swap_player(player), move_history) # loss

        move_history.append((player, move[m], board.copy()))
        game_status = make_move( player, m, move[m], board)
    print_board( board )
    print()
    if game_status == WIN:
        return (player, move_history)
    elif game_status == DRAW:
        return (DRAW, move_history)
    raise Exception("We shouldn't be here")


def main():

    wins = [0,0]
    p1 = get_net(player=1)
    p2 = get_net(player=2)
    game_status = 0
    while game_status != DRAW:
        print(wins, end="\r")
        game_status, move_history = game(p1.target_net, p2.target_net)
        # print("status", game_status)
        if game_status == 1:
            wins[0] += 1
            wins[1] =0 
            update_training_gameover(p1, move_history[::-1], 1, 1.0)
            update_training_gameover(p2, move_history[::-1], 0, 1.0)
        elif game_status == 2:
            wins[0] = 0
            wins[1] += 1 
            update_training_gameover(p1, move_history[::-1], 0, 1.0)
            update_training_gameover(p2, move_history[::-1], 1, 1.0)
        else:
            update_training_gameover(p1, move_history[::-1], 0.5, 1.0)
            update_training_gameover(p2, move_history[::-1], 0.5, 1.0)
    torch.save(p1.target_net.state_dict(), './32p1.pth')
    torch.save(p2.target_net.state_dict(), './32p2.pth')



#**********************************************************
#   Print the board
#
def print_board( bd ):
    sb = '.XO'
    print('|',sb[bd[1]],sb[bd[2]],sb[bd[3]],'|')
    print('|',sb[bd[4]],sb[bd[5]],sb[bd[6]],'|')
    print('|',sb[bd[7]],sb[bd[8]],sb[bd[9]],'|')


#**********************************************************
#   Make specified move on the board and return game status
#
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

#**********************************************************
#   Return True if the board is full
#
def full_board( board ):
    b = 1
    while b <= 9 and board[b] != EMPTY:
        b += 1
    return( b == 10 )

#**********************************************************
#   Return True if game won by player p on board bd[]
#
def game_won( p, bd ):
    return(  ( bd[1] == p and bd[2] == p and bd[3] == p )
           or( bd[4] == p and bd[5] == p and bd[6] == p )
           or( bd[7] == p and bd[8] == p and bd[9] == p )
           or( bd[1] == p and bd[4] == p and bd[7] == p )
           or( bd[2] == p and bd[5] == p and bd[8] == p )
           or( bd[3] == p and bd[6] == p and bd[9] == p )
           or( bd[1] == p and bd[5] == p and bd[9] == p )
           or( bd[3] == p and bd[5] == p and bd[7] == p ))


# https://nestedsoftware.com/2019/12/27/tic-tac-toe-with-a-neural-network-1fjn.206436.html
def update_training_gameover(net_context, move_history, game_result_reward, discount_factor):

    # move history is in reverse-chronological order - last to first
    next_move = move_history[0]
    next_move_index = next_move[1]
    next_position = next_move[2]

    print("here")
    backpropagate(net_context, next_position, next_move_index, game_result_reward-1)
    print("there")
    for i in range(1, len(move_history)):
        next_q_values = get_q_values(next_position, net_context.target_net)
        qv = torch.max(next_q_values).item()

        backpropagate(net_context, board_history[i], move_history[i], discount_factor * qv)

        next_position = board_history[i]

    net_context.target_net.load_state_dict(net_context.policy_net.state_dict())

def convert_to_tensor(bd, player):
    out = torch.from_numpy(np.append(bd[1:]/2, player % 2)).float() 
    out = out.unsqueeze(0)
    return out


def get_q_values(bd: np.array, model: NeuralNetwork):
    inputs = convert_to_tensor(bd)
    outputs = model(inputs).detach()
    return outputs[0]

def get_illegal_move_indexes(bd: np.array):
    return [i for i, x in enumerate(bd) if x != EMPTY or i == 0]

def backpropagate(net_context, position, move_index, target_value):
    net_context.optimizer.zero_grad()
    output = net_context.policy_net(convert_to_tensor(position))
    target = output.clone().detach()[0]
    target[move_index] = target_value
    illegal_move_indexes = get_illegal_move_indexes(position)
    for mi in illegal_move_indexes:
        print(target)
        print(mi)
        target[mi] = 0

    loss = net_context.loss_function(output, target)
    loss.backward()
    net_context.optimizer.step()

if __name__ == '__main__':
    main()
