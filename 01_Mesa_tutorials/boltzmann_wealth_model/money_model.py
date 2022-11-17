import mesa

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N*sum(x))
    return 1+(1/N) -2 *B


class MoneyAgent(mesa.Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id,model)
        self.wealth = 1

    def move(self):
        neighbor_cells = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False
        )
        new_position = self.random.choice(neighbor_cells)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        # if there is another agent in the same cell
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

    def step(self):
        # the agent steps
        self.move()
        if self.wealth > 0:
            self.give_money()
        #print(f'agent {str(self.unique_id)} has wealth {self.wealth}.')


class MoneyModel(mesa.Model):

    def __init__(self, N, width, height):
        super().__init__(N)
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, torus=True)
        self.schedule = mesa.time.RandomActivation(self)
        # create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

            # add agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))

        self.datacollector = mesa.DataCollector(
            model_reporters={'Gini' : compute_gini}, agent_reporters={'Wealth' : 'wealth'}
        )

    def step(self):
        # advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()