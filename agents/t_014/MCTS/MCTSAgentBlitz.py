from agents.t_014.MCTS.MCTSState import MCTSState
from agents.t_014.MCTS.MonteCarloTreeSearchBlitz import MonteCarloTreeSearchBlitz
from template import Agent


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
    
    def SelectAction(self, actions, game_state):
        agent = MonteCarloTreeSearchBlitz(self.id, MCTSState(self.id, game_state))
        return agent.MCTSearch()
