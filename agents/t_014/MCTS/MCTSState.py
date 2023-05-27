class MCTSState:
    def __init__(self, player_id, azul_state, previous_move=None):
        self.player_id = player_id
        self.azul_state = azul_state
        self.previous_move = previous_move
