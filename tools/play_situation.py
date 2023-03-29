import time


class PlaySituation:
    def __init__(self, model):
        self.model = model

    def play(self):
        self.model.initialize_game(render="human")
        self.model.reset_agents()
        end = False
        i = 0
        while end is False:
            current_agent = self.model.agents[i % 2]["name"]
            self.model.env.agent_selection = current_agent
            state = self.model.env.observe(current_agent)
            action = self.model.get_action(state)
            self.model.env.step(action)
            state, reward, termination, truncation, info = self.model.env.last()
            end = termination or truncation
            i += 1
            time.sleep(0.3)
        self.env.close()
