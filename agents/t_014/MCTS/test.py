# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import random
import time
import math
import numpy as np
import Azul.azul_utils as utils

from scipy import signal
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule
from agents.t_014.MCTS.MCTSNode import MCTSNode
from agents.t_014.MCTS.MCTSState import MCTSState

THINK_TIME = 0.5
NUM_PLAYERS = 2


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


# Defines this agent.
# noinspection DuplicatedCode
class MonteCarloTreeSearchLines:

    def __init__(self, _id, initial_state, time_limit=THINK_TIME, exploration_constant=1, discount_factor=0.8):
        self.id = _id  # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS)  # Agent stores an instance of GameRule, from which to obtain functions.
        self.time_limit = time_limit  # Maximum computational run time for each move
        self.root = MCTSNode(initial_state)  # Initialise the node using the initial board state
        self.exploration_constant = exploration_constant  # Exploration constant for UCB1 selection
        self.discount_factor = discount_factor  # Discount factor during backpropagation node updates
        self.gaussian_kernel = self.gaussianFilter(1.5)  # Create the Gaussian distribution for central tile weighing
        self.early_game = 2

    def MCTSearch(self):
        start_time = time.time()
        self.expandNode()
        # Iterate for given time limit
        while THINK_TIME > (time.time() - start_time):
            # Select a node using Multi Armed Robber with UCB1, specifying the exploration constant C value
            node = self.MABBestMove(self.root)
            # Run simulation to get a score of the node
            scores = self.simulation(node.state)
            # Backpropagation of results
            self.backpropagation(node, scores)
        # Return best move found in given time
        return self.chooseBestMove(self.root.children)

    def expandNode(self):
        # Get all the legal moves for the player for the given state
        possible_moves = self.game_rule.getLegalActions(self.root.state.azul_state, self.id)
        # Since there are only 2 players, id of the opposing player is determined
        opposing_player_id = 1 - self.id
        for move in possible_moves:
            # Generate all the children state, creating a new child node for each possible move
            parent_azul_state = deepcopy(self.root.state.azul_state)
            child_azul_state = self.game_rule.generateSuccessor(parent_azul_state, move, self.id)
            child_state = MCTSState(opposing_player_id, child_azul_state, move)
            self.root.children.append(MCTSNode(child_state, self.root))

    def MABBestMove(self, node):
        best_score = float('-inf')
        best_child = []
        for child in node.children:
            # New node found, explore this node first
            if child.num_visited == 0:
                return child
            else:
                # UCB1 formula for move score with given exploration constant
                move_score = child.q_score + 2 * self.exploration_constant * \
                             math.sqrt(math.log(node.num_visited / child.num_visited))
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
        lines_state_0 = deepcopy(state_simulation.agents[0].lines_number)
        lines_state_1 = deepcopy(state_simulation.agents[1].lines_number)
        current_player = self.id
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
        scores = [self.calculateBonus(state_simulation, lines_state_0, 0),
                  self.calculateBonus(state_simulation, lines_state_1, 1)]
        return scores

    def backpropagation(self, node, scores):
        # Backpropagation of node score until root node
        while node is not None and scores is not None:
            node.num_visited += 1
            # Calculate the player scores using a discount factor and depth of the backpropagation
            node.player_scores = (node.player_scores[0] +
                                  (self.discount_factor * scores[0] -
                                   node.player_scores[0]) / node.num_visited,
                                  node.player_scores[1] +
                                  (self.discount_factor * scores[1] -
                                   node.player_scores[1]) / node.num_visited)
            # Take q score as the difference of agent score and opponent score
            node.q_score = node.q_score + \
                           (self.discount_factor * (node.player_scores[self.id] - node.player_scores[1 - self.id]) - node.q_score)/node.num_visited
            # Move up a node to the parent node
            node = node.parent_node

    def chooseBestMove(self, child_nodes):
        # Choose the move with the largest q score
        current_max = float('-inf')
        current_best_move = None
        for node in child_nodes:
            if node.q_score > current_max:
                current_max = node.q_score
                current_best_move = node.state.previous_move
        if current_best_move is not None:
            return current_best_move
        else:
            # Randomly return a legal move
            return random.choice(self.game_rule.getLegalActions(self.root, self.id))

    def calculateBonus(self, azul_state, previous_lines, player_id):
        game_ending = False
        if azul_state.agents[player_id].GetCompletedRows() > 0 or \
                azul_state.agents[1 - player_id].GetCompletedRows() > 0:
            game_ending = True
        if not game_ending:
            column_bonus, column_rows = self.columnBonus(azul_state, player_id, game_ending)
            set_bonus, set_rows = self.setBonus(azul_state, player_id, game_ending)
            excluded_rows = column_rows + set_rows
            return \
                self.linesBonus(azul_state, player_id, previous_lines) + \
                azul_state.agents[player_id].ScoreRound()[0] + \
                1.5 * column_bonus + \
                1.5 * set_bonus + \
                1.5 * self.centreBonus(azul_state, player_id) + \
                self.rowBonus(azul_state, player_id) - \
                self.unfinishedPenalty(azul_state, player_id, excluded_rows, game_ending)
        else:
            azul_state.agents[player_id].EndOfGameScore()
            return azul_state.agents[player_id].score

    @staticmethod
    # Intermediate bonus for completing columns
    def columnBonus(azul_state, player_id, game_ending):
        player_board = azul_state.agents[player_id]
        column_reward = 0
        column_rows = []
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

            # if contiguous_count == 3 and not game_ending:
            #     column_reward += 1
            if contiguous_count == 4 and not game_ending:
                column_reward += 3
                # if player_board.lines_tile[missing_line] == -1:
                #     column_reward += 3
                # # If a column already has 4 tiles penalty for placing the wrong tile on the remaining row
                # elif player_board.lines_tile[missing_line] != -1 and \
                #         player_board.lines_tile[missing_line] != missing_tile:
                #     column_reward += 0
                # # If a column already has 4 tiles reward placing the correct tile on the remaining row
                if player_board.lines_tile[missing_line] == missing_tile:
                    column_reward += (player_board.lines_number[missing_line])/2
                    column_rows.append(missing_line)
            if contiguous_count == player_board.GRID_SIZE:
                column_reward += 7
        return column_reward, column_rows

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
        set_rows = []
        for tile in utils.Tile:
            if player_board.number_of[tile] == 4 and not game_ending:
                set_reward += 4.3
                for i in range(player_board.GRID_SIZE):
                #     # If a set already has 4 tiles penalty for placing the wrong tile on the remaining row
                #     if player_board.grid_state[i][int(player_board.grid_scheme[i][tile])] == 0:
                #         if player_board.lines_tile[i] == -1:
                #             set_reward += 4.3
                #         # If a set already has 4 tiles penalise placing a wrong tile on the remaining row
                #         elif player_board.lines_tile[i] != -1 and player_board.lines_tile[i] != tile:
                #             set_reward += 1.3
                #         # If a set already has 4 tiles reward placing the correct tile on the remaining row
                    if player_board.lines_tile[i] == tile:
                        set_rows.append(i)
                        set_reward += (player_board.lines_number[i])/2
                #             # print("---------------------")
                #             # print(i)
                #             # print(tile)
                #             # print(5 + (player_board.lines_number[i])/2)
                #             # print("---------------------")
            if player_board.number_of[tile] == player_board.GRID_SIZE:
                set_reward += 10
        return set_reward, set_rows

    @staticmethod
    # Creating a gaussian filter of the board size
    def gaussianFilter(std, normalised=False):
        gaussian_1d = signal.gaussian(5, std)
        gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
        if normalised:
            gaussian_2d /= (2*np.pi*(std**2))
        return gaussian_2d

    # Centre bonus
    def centreBonus(self, azul_state, player_id):
        player_board = azul_state.agents[player_id]
        centre_bonus = 0
        for i in range(player_board.GRID_SIZE):
            for j in range(player_board.GRID_SIZE):
                centre_bonus += self.gaussian_kernel[i][j] * player_board.grid_state[i][j]
        return centre_bonus

    # Incurred penalty for unfinished lines and rows if they are not completing a column or a set
    def unfinishedPenalty(self, azul_state, player_id, excluded_rows, game_ending):
        player_board = azul_state.agents[player_id]
        unfinished_penalty = 0
        # Penalising rows for having leftover tiles or unfinished lines if the game is not ending
        if not game_ending:
            # Row 4
            if 3 not in excluded_rows:
                if player_board.lines_number[3] == 1:
                    unfinished_penalty += 1
                elif player_board.lines_number[3] == 2:
                    unfinished_penalty += 0.5
            # Row 5
            if 4 not in excluded_rows:
                if player_board.lines_number[4] == 1:
                    unfinished_penalty += 1.5
                elif player_board.lines_number[4] == 2:
                    unfinished_penalty += 1

            # Penalising unfinished rows in a round after early game
            for i in range(1, player_board.GRID_SIZE):
                if player_board.lines_number[i] > 0:
                    unfinished_penalty += 0.8

        return unfinished_penalty

    @staticmethod
    def startBonus(azul_state, player_id, game_ending):
        start_bonus = 0
        if azul_state.next_first_agent == player_id and not game_ending:
            start_bonus += 0.1
        return start_bonus

    def linesBonus(self, azul_state, player_id, previous_lines):
        lines_bonus = 0
        ending_lines = azul_state.agents[player_id].lines_number
        if ending_lines[2] == 3:
            lines_bonus += 1.5
        # elif previous_lines[2] == 0 and ending_lines[2] == 2:
        #     lines_bonus += 0.25

        # if previous_lines[3] == 0 and ending_lines[3] == 3:
        #     lines_bonus += 0.8
        if 0 < previous_lines[3] < ending_lines[3]:
            lines_bonus += 0.7
        elif ending_lines[3] == 4:
            lines_bonus += 2.5

        # if previous_lines[4] == 0 and ending_lines[4] == 3:
        #     lines_bonus += 0.7
        # elif previous_lines[4] == 0 and ending_lines[4] == 3:
        #     lines_bonus += 1.2
        if 0 < previous_lines[4] < ending_lines[4]:
            lines_bonus += 0.7
        elif ending_lines[4] == 5:
            lines_bonus += 3.5

        return 0.9 ** len(azul_state.agents[player_id].agent_trace.round_scores) * lines_bonus

# END FILE -----------------------------------------------------------------------------------------------------------#