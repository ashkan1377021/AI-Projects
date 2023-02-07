# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        """*** YOUR CODE HERE ***"""
        for i in range(self.iterations):
            temp = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    temp[state] = 0
                else:
                    Q_max = -float('inf')
                    possible_actions = self.mdp.getPossibleActions(state)
                    for possible_action in possible_actions:
                        Q_max = max(Q_max, self.computeQValueFromValues(state, possible_action))
                    if Q_max > -float('inf'):
                        temp[state] = Q_max
            self.values = temp

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_value = 0
        for prob_state in self.mdp.getTransitionStatesAndProbs(state, action):
            Q_value += prob_state[1] * (
                    self.mdp.getReward(state, action, prob_state[0]) + self.discount * self.values[prob_state[0]])
        return Q_value
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.mdp.isTerminal(state):
            possible_actions = self.mdp.getPossibleActions(state)
            Q_max = -float('inf')
            best_action = None
            for possible_action in possible_actions:
                current_Q = self.computeQValueFromValues(state, possible_action)
                if current_Q > Q_max:
                    Q_max = current_Q
                    best_action = possible_action
            return best_action

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            if self.mdp.isTerminal(states[i % len(states)]):
                continue
            Q_max = -float('inf')
            for possible_action in self.mdp.getPossibleActions(states[i % len(states)]):
                Q_max = max(Q_max, self.computeQValueFromValues(states[i % len(states)], possible_action))
            if Q_max > -float('inf'):
                self.values[states[i % len(states)]] = Q_max


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """*** YOUR CODE HERE ***"""
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = self.findPredecessors(state)

        priorityQueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                Q_max = -float('inf')
                for possible_action in self.mdp.getPossibleActions(state):
                    Q_max = max(Q_max, self.computeQValueFromValues(state, possible_action))
                diff = abs(self.values[state] - Q_max)
                priorityQueue.update(state, -diff)

        for i in range(self.iterations):
            if not priorityQueue.isEmpty():
                state = priorityQueue.pop()

                if not self.mdp.isTerminal(state):
                    possible_actions = self.mdp.getPossibleActions(state)
                    Q_max= -float('inf')
                    for possible_action in possible_actions:
                        Q_max = max(Q_max,self.computeQValueFromValues(state,possible_action))
                    if Q_max > -float('inf'):
                        self.values[state] = Q_max

                    for predecessor in list(predecessors[state]):
                        if not self.mdp.isTerminal(predecessor):
                            possible_actions = self.mdp.getPossibleActions(predecessor)
                            Q_max = - float('inf')
                            for possible_action in possible_actions:
                                Q_max = max(Q_max, self.computeQValueFromValues(predecessor, possible_action))
                            diff = abs(self.values[predecessor] - Q_max)
                            if diff > self.theta:
                                priorityQueue.update(predecessor, -diff)
            else:
                break

    def findPredecessors(self, currentState):

        predecessors = set()

        if not self.mdp.isTerminal(currentState):
            for state in self.mdp.getStates():
                possible_actions = self.mdp.getPossibleActions(state)
                for possible_action in possible_actions:
                    prob_states = self.mdp.getTransitionStatesAndProbs(state, possible_action)
                    for prob_state in prob_states:
                        if prob_state[0] == currentState and prob_state[1] > 0:
                            predecessors.add(state)

        return predecessors
