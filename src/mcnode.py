class McNode:
    def __init__(self, state, curr_board, parent=None, visits=0):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = visits
        self.curr_board = curr_board
    def __eq__(self, other: object) -> bool:
        return (self.state == other.state) and (self.curr_board == other.curr_board)
    def __str__(self) -> str:
        return f"wins = {self.wins} visits = {self.visits} curr_board = {self.curr_board}\n children = {self.children}"
    def visited(self):
        self.visit += 1
    def won(self):
        self.wins += 1