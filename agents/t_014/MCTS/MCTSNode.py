class MCTSNode:
    def __init__(self, state, parent_node=None):
        self.state = state
        self.parent_node = parent_node
        self.children = []
        self.num_visited = 0
        self.player_scores = (0, 0)
        self.q_score = 0
