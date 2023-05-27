# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import random
import time
import math
import Azul.azul_utils as utils

from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule
from agents.t_014.MCTS.MCTSNode import MCTSNode
from agents.t_014.MCTS.MCTSState import MCTSState

THINK_TIME = 0.97
NUM_PLAYERS = 2


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


# Defines this agent.
# noinspection DuplicatedCode
class MonteCarloTreeSearchBasic:

    def __init__(self, _id, initial_state, time_limit=THINK_TIME, exploration_constant=1/math.sqrt(2), discount_factor=0.9):
        self.id = _id  # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS)  # Agent stores an instance of GameRule, from which to obtain functions.
        self.time_limit = time_limit  # Maximum computational run time for each move
        self.root = MCTSNode(initial_state)  # Initialise the node using the initial board state
        self.exploration_constant = exploration_constant  # Exploration constant for UCB1 selection
        self.discount_factor = discount_factor  # Discount factor during backpropagation node updates
        self.EARLY_GAME = 2  # Early game rounds
        self.COLUMN_BONUS = [0, 0.5, 2, 2.7, 3.5]
        self.ROW_BONUS = [0, 0.8, 1.4, 0.8, 0.1]

    def MCTSearch(self):
        start_time = time.time()
        # Iterate for given time limit
        while THINK_TIME > (time.time() - start_time):
            # Select a node using Multi Armed Robber with UCB1, specifying the exploration constant C value
            node = self.root
            while len(node.children) > 0 and node.num_visited > 0:
                node = self.MABBestMove(node)
            if node.state.azul_state.TilesRemaining():
                self.expandNode(node)
                if len(node.children) > 0:
                    node = random.choice(node.children)
            # Run simulation to get a score of the node
            scores = self.simulation(node.state)
            # Backpropagation of results
            self.backpropagation(node, scores)
        # Return best move found in given time
        return self.chooseBestMove(self.root.children)

    def expandNode(self, node):
        # Get all the legal moves for the player for the given state
        player_id = node.state.player_id
        possible_moves = self.pruneMoves(self.game_rule.getLegalActions(node.state.azul_state, player_id))
        # Since there are only 2 players, id of the opposing player is determined
        opposing_player_id = 1 - player_id
        for move in possible_moves:
            # Generate all the children state, creating a new child node for each possible move
            parent_azul_state = deepcopy(node.state.azul_state)
            child_azul_state = self.game_rule.generateSuccessor(parent_azul_state, move, player_id)
            child_state = MCTSState(opposing_player_id, child_azul_state, move)
            node.children.append(MCTSNode(child_state, node))

    def MABBestMove(self, node):
        best_score = float('-inf')
        best_child = []
        for child in node.children:
            # New node found, explore this node first
            if child.num_visited == 0:
                return child
            else:
                # UCB1 formula for move score with given exploration constant
                move_score = (child.player_scores[self.id] - child.player_scores[1 - self.id]) + \
                             2 * self.exploration_constant * math.sqrt(math.log(node.num_visited / child.num_visited))
                # Better node with better score to explore is found
                if move_score > best_score:
                    best_score = move_score
                    best_child = [child]
                # Append to the list if the best score is the same
                elif move_score == best_score:
                    best_child.append(child)
        # Randomly return one of the best nodes if there are multiple nodes with same score
        return random.choice(best_child)

    def simulation(self, state):
        # Make a copy of the azul board state to run a simulation on
        state_simulation = deepcopy(state.azul_state)
        board_state_0 = deepcopy(state_simulation.agents[0].grid_state)
        board_state_1 = deepcopy(state_simulation.agents[1].grid_state)
        lines_state_0 = deepcopy(state_simulation.agents[0].lines_number)
        lines_state_1 = deepcopy(state_simulation.agents[1].lines_number)
        current_player = state.player_id
        num_moves = 0
        # Continue iterating if there are tiles remaining on the field (round is not complete)
        while state_simulation.TilesRemaining():
            # Randomly select one of the legal moves
            possible_moves = self.game_rule.getLegalActions(state_simulation, current_player)
            chosen_move = random.choice(possible_moves)
            # Iterate the simulated state using the random valid move
            state_simulation = self.game_rule.generateSuccessor(state_simulation, chosen_move, current_player)
            # Switch to the other player to play a move and add to move count
            current_player = 1 - current_player
            num_moves += 1
        # Aggregate the scores accumulated at the end of the round along with intermediate bonuses and penalties
        scores = [self.calculateBonus(state_simulation, lines_state_0, board_state_0, 0),
                  self.calculateBonus(state_simulation, lines_state_1, board_state_1, 1)]
        return scores

    def backpropagation(self, node, scores):
        # Backpropagation of node score until root node
        while node is not None and scores is not None:
            node.num_visited += 1
            # Discount reward
            scores = [score * self.discount_factor for score in scores]
            # Calculate the player scores using a discount factor and depth of the backpropagation
            node.player_scores = \
                (node.player_scores[0] +
                 (self.discount_factor * scores[0] - node.player_scores[0]) / node.num_visited,
                 node.player_scores[1] +
                 (self.discount_factor * scores[1] - node.player_scores[1]) / node.num_visited)
            # Move up a node to the parent node
            node = node.parent_node

    def chooseBestMove(self, child_nodes):
        # Choose the move with the largest q score
        current_max = float('-inf')
        current_best_move = None
        for node in child_nodes:
            if node.player_scores[self.id] - node.player_scores[1 - self.id] > current_max:
                current_max = node.player_scores[self.id] - node.player_scores[1 - self.id]
                current_best_move = node.state.previous_move
        if current_best_move is not None:
            return current_best_move
        else:
            # Randomly return a legal move
            return random.choice(self.game_rule.getLegalActions(self.root, self.id))

    def calculateBonus(self, azul_state, previous_lines, previous_board, player_id):
        game_ending = False
        if azul_state.agents[player_id].GetCompletedRows() > 0 or \
                azul_state.agents[1 - player_id].GetCompletedRows() > 0:
            game_ending = True
        if not game_ending:
            score = azul_state.agents[player_id].ScoreRound()[0] + \
                    self.columnBonus(azul_state, player_id, game_ending) + \
                    self.setBonus(azul_state, player_id, game_ending) + \
                    self.rowBonus(azul_state, player_id) + \
                    self.startBonus(azul_state, player_id, game_ending) + \
                    - self.unfinishedPenalty(azul_state, player_id, game_ending)
            return score
        else:
            return azul_state.agents[player_id].ScoreRound()[0] + azul_state.agents[player_id].EndOfGameScore()

    @staticmethod
    # Intermediate bonus for completing columns
    def columnBonus(azul_state, player_id, game_ending):
        player_board = azul_state.agents[player_id]
        column_reward = 0
        for i in range(player_board.GRID_SIZE):
            contiguous_count = 0
            missing_line = None
            missing_tile = None
            for j in range(player_board.GRID_SIZE):
                if player_board.grid_state[j][i] == 1:
                    contiguous_count += 1
                for tile in utils.Tile:
                    if player_board.grid_scheme[j][tile] == i:
                        missing_tile = tile
                        missing_line = j
            # if contiguous_count == 4 and not game_ending:
            #     if player_board.lines_tile[missing_line] == missing_tile:
            #         column_reward += 0.5 * (player_board.lines_number[i])
            if contiguous_count == player_board.GRID_SIZE:
                column_reward += 7
        return column_reward

    @staticmethod
    # Bonus for completing rows
    def rowBonus(azul_state, player_id):
        player_board = azul_state.agents[player_id]
        row_reward = 0
        for i in range(player_board.GRID_SIZE):
            contiguous_count = 0
            for j in range(player_board.GRID_SIZE):
                if player_board.grid_state[i][j] == 1:
                    contiguous_count += 1
            if contiguous_count == player_board.GRID_SIZE:
                row_reward += 2
        return row_reward

    @staticmethod
    # Intermediate bonus for completing sets
    def setBonus(azul_state, player_id, game_ending):
        player_board = azul_state.agents[player_id]
        set_reward = 0
        for tile in utils.Tile:
            # if player_board.number_of[tile] == 4 and not game_ending:
            #     for i in range(player_board.GRID_SIZE):
            #         # If a set already has 4 tiles reward for placing right tile on the remaining row
            #         if player_board.grid_state[i][int(player_board.grid_scheme[i][tile])] == 0:
            #             if player_board.lines_tile[i] == tile:
            #                 set_reward += 0.5 * (player_board.lines_number[i])
            if player_board.number_of[tile] == player_board.GRID_SIZE:
                set_reward += 10
        return set_reward

    # Incurred penalty for unfinished lines and rows if they are not completing a column or a set
    @staticmethod
    def unfinishedPenalty(azul_state, player_id, game_ending):
        player_board = azul_state.agents[player_id]
        unfinished_penalty = 0
        ending_lines = azul_state.agents[player_id].lines_number
        ending_tiles = azul_state.agents[player_id].lines_tile
        # Penalising rows for having leftover tiles or unfinished lines if the game is not ending
        if not game_ending:
            # Row 4
            if player_board.lines_number[3] == 1:
                unfinished_penalty += 1.5
            elif player_board.lines_number[3] == 2:
                unfinished_penalty += 0.5
            # Row 5
            if player_board.lines_number[4] == 1:
                unfinished_penalty += 2
            elif player_board.lines_number[4] == 2:
                unfinished_penalty += 1
            # Unfinished rows, no penalty on round 1 since no round points accrued
            if len(azul_state.agents[player_id].agent_trace.round_scores) > 1:
                for i in range(1, player_board.GRID_SIZE):
                    if player_board.lines_number[i] > 0:
                        unfinished_penalty += 0.5
        return unfinished_penalty

    @staticmethod
    def startBonus(azul_state, player_id, game_ending):
        start_bonus = 0
        if azul_state.next_first_agent == player_id and not game_ending and \
                len(azul_state.agents[player_id].agent_trace.round_scores) > 1:
            start_bonus += 0.6
        return start_bonus

    @staticmethod
    def pruneMoves(moves):
        good_moves = []
        for move in moves:
            if len(moves) > 40:
                if move[2].pattern_line_dest >= 3 and move[2].num_to_pattern_line == 1:
                    continue
            if len(moves) > 20:
                if move[2].num_to_floor_line == move[2].number:
                    continue
            if len(moves) > 5:
                if move[2].num_to_floor_line == move[2].number and move[2].number >= 3:
                    continue
            good_moves.append(move)
        return good_moves

# END FILE -----------------------------------------------------------------------------------------------------------#
