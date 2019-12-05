# grid_world.py
import random, operator

def argmax(seq, fn):
  best = seq[0];
  best_score = fn(best)
  for x in seq:
    x_score = fn(x)
    if x_score > best_score:
      best, best_score = x, x_score

  return best

def vector_add(a, b):
  return tuple(map(operator.add, a, b))

orientations = [(1,0), (0, 1), (-1,0), (0, -1)]

def turn_right(orientation):
  return orientations[orientations.index(orientation) - 1]

def turn_left(orientation):
  return orientations[(orientations.index(orientation) + 1) % len(orientations)]

def isnumber(x):
  return hasattr(x, '__int__')

class MDP:
  def __init__(self, init_pos, actlist, terminals, transitions={}, states=None, gamma=0.99):
    if not (0 < gamma <=1):
      raise ValueError("MDP should have 0 < gamma <=
        1 values")
    if states:
      self.states = states
    else:
      self.states = set()
    self.init_pos = init_pos
    self.actlist = actlist
    self.terminals = terminals
    self.transitions = transitions
    self.gamma = gamma
    self.reward = {}

  def R(self, state):
    return self.reward[state]

  def T(self, state, action):
    if(self.transitions == {}):
      raise ValueError("Transition model is missing")
    else:
      return self.transitions[state][action]

  def actions(self, state):
    if state in self.terminals:
      return [None]
    
    return self.actlist

class GridMDP(MDP):
  def __init__(self, grid, terminals, init_pos=(0, 0), gamma=0.99):
    grid.reverse()

    MDP.__init__(self, init_pos, actlist=orientations, terminals=terminals, gamma=gamma)
    self.grid = grid
    self.rows = len(grid)
    self.cols = len(grid[0])
    for x in range(self.cols):
      for y in range(self.rows):
        self.reward[x, y] = grid[y][x]
        if grid[y][x] is not None:
          self.states.add((x, y))

  def T(self, state, action):
    if action is None:
      return [(0.0, state)]

    return [(0.8, self.go(state, action)),
            (0.1, self.go(state, turn_right(action))), 
            (0.1, self.go(state, turn_left(action)))]

  def go(self, state, direction):
    state1 = vector_add(state, direction)
    return state1 if state1 in self.states else state

  def to_grid(self, mapping):
    return list(reversed([[mapping.get((x, y), None) for x in range(self.cols)] for y in range(self.rows)]))

  def to_arrows(self, policy):
    chars = {(1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
    return self.to_grid({s: chars[a] for (s, a) in policy.items()})

def value_iteration(mdp, epsilon=0.001):
  STSN = {s: 0 for s in mdp.states}
  R, T, gamma = mdp.R, mdp.T, mdp.gamma
  while True:
    STS = STSN.copy()
    delta = 0
    for s in mdp.states:
      STSN[s] = R(s) + gamma * max([sum([p * STS[s1] for (p, s1) in T(s, a)]) 
        for a in mdp.actions(s)])
      delta = max(delta, abs(STSN[s] -STS[s]))

    if delta < epsilon * (1 - gamma) / gamma:
      return STS

def best_policy(mdp, STS):
  pi = {}
  for s in mdp.states:
    pi[s] = argmax(mdp.actions(s), lambda a: expected_utility(a, s, STS, mdp))
    print(pi)
    return pi

def expected_utility(a, s, STS, mdp):
  return sum([p * STS[s1] for (p, s1) in mdp.T(s, a)])

def policy_iteration(mdp):
  STS == {s: 0 for s in mdp.states}
  pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
  while True:
    STS = policy_evaluation(pi, STS, mdp)
    unchanged = True
    for s in mdp.states:
      a = argmax(mdp.actions(s), lambda a: expected_utility(a, a, STS, mdp))
      if a != pi[s]:
        pi[s] = a
        unchanged = False
    if unchanged:
      return pi

def policy_evaluation(pi, STS, mdp, k=20):
  R, T, gamma = mdp.R, mdp.T, mdp.gamma
  for i in range(k):
    for s in mdp.states:
      STS[s] = R(s) + gamma * sum([p * STS[s1] for (p, s1) in T(s, pi[s])])

  return STS

def print_table(table, header=None, sep='   ', numfmt='{}'):
  justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]
  if header:
    table.insert(0, header)

  table = [[numfmt.format(x) if isnumber(x) else x for x in row] for row in table]

  sizes = list(map(lambda seq: max(map(len, seq)), list(zip(*[map(str, row) for row in table]))))

  for row in table:
    print(sep.join(getattr(str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))

sequential_decision_enviroment = GridMDP([[-0.02, -0.02, -0.02, + 1],

                                          [-0.02, None, -0.02, -1],
                                          [-0.02, -0.02, -0.02,-0.02]], terminals=[(3, 2), (3, 1)])


value_iter = best_policy(sequential_decision_enviroment, value_iteration(sequential_decision_enviroment, .01))
print(value_iter)
print("\n Optimal Policy based on Value Iteration\n")
print_table(sequential_decision_enviroment.to_arrows(value_iter))


