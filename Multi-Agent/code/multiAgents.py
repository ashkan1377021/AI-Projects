# multiAgents.py
# --------------
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


import random

import util
from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        currentFoods = currentGameState.getFood().asList()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        evaluation = 0
        # calculate distance of the nearest food (among current foods )from new position
        nearestFood = float('inf')
        for food in currentFoods:
            temp = manhattanDistance(newPos, food)
            if temp < nearestFood:
                nearestFood = temp
        evaluation -= nearestFood
        # calculate distance of the nearest(among active ghosts)ghost from new position
        nearestGhost = float('inf')
        for ghost in newGhostStates:
            if not ghost.scaredTimer > 0:
                temp = manhattanDistance(newPos, ghost.getPosition())
                if temp < nearestGhost:
                    nearestGhost = temp
        # checks that this position can lead to game over or not
        if nearestGhost <= 1:
            return -float('inf')
        return evaluation


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
       Your minimax agent (question 2)
       """

    def getAction(self, gameState):
        def minimizer(state, index, depth):
            legalActions = state.getLegalActions(index)
            if not legalActions:
                return self.evaluationFunction(state)
            if index == state.getNumAgents() - 1:
                return min(maximizer(state.generateSuccessor(index, action), depth) for action in legalActions)
            else:
                return min(minimizer(state.generateSuccessor(index, action), index + 1, depth) for action in
                           legalActions)

        def maximizer(state, depth):
            legalActions = state.getLegalActions(0)
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)
            return max(minimizer(state.generateSuccessor(0, action), 1, depth + 1) for action in legalActions)

        legals = gameState.getLegalActions(0)
        values = []
        for i in range(len(legals)):
            values.append(minimizer(gameState.generateSuccessor(0, legals[i]), 1, 1))
        maxValue = max(values)
        bestIndices = [index for index in range(len(values)) if values[index] == maxValue]
        return legals[random.choice(bestIndices)]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        def minimizer(state, index, depth, alpha, beta):
            legalActions = state.getLegalActions(index)
            if not legalActions:
                return self.evaluationFunction(state)
            minValue = float('inf')
            if index == state.getNumAgents() - 1:
                for action in legalActions:
                    temp = maximizer(state.generateSuccessor(index, action), depth, alpha, beta)
                    if temp < minValue:
                        minValue = temp
                    beta = min(beta, minValue)
                    if beta < alpha:
                        return minValue
            else:
                for action in legalActions:
                    temp = minimizer(state.generateSuccessor(index, action), index + 1, depth, alpha, beta)
                    if temp < minValue:
                        minValue = temp
                    beta = min(beta, minValue)
                    if beta < alpha:
                        return minValue

            return minValue

        def maximizer(state, depth, alpha, beta):
            legalActions = state.getLegalActions(0)
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)
            maxValue = -float('inf')
            for action in legalActions:
                temp = minimizer(state.generateSuccessor(0, action), 1, depth + 1, alpha, beta)
                if temp > maxValue:
                    maxValue = temp
                alpha = max(alpha, maxValue)
                if beta < alpha:
                    return maxValue
            return maxValue

        legals = gameState.getLegalActions(0)
        values = []
        a = -float('inf')
        b = float('inf')
        for i in range(len(legals)):
            temp = minimizer(gameState.generateSuccessor(0, legals[i]), 1, 1, a, b)
            values.append(temp)
            if temp > a:
                a = temp
        bestIndices = [index for index in range(len(values)) if values[index] == a]
        return legals[random.choice(bestIndices)]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        def maximizer(state, depth):
            legalActions = state.getLegalActions(0)
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            return max(expect(state.generateSuccessor(0, action), 1, depth + 1) for action in legalActions)

        def expect(state, index, depth):
            legalActions = state.getLegalActions(index)
            if not legalActions:
                return self.evaluationFunction(state)
            value = 0
            if index == state.getNumAgents() - 1:
                for action in legalActions:
                    value += maximizer(state.generateSuccessor(index, action), depth) * (1.0 / len(legalActions))
            else:
                for action in legalActions:
                    value += expect(state.generateSuccessor(index, action), index + 1, depth) * (
                            1.0 / len(legalActions))
            return value

        legals = gameState.getLegalActions(0)
        values = []
        for i in range(len(legals)):
            values.append(expect(gameState.generateSuccessor(0, legals[i]), 1, 1))
        maxValue = max(values)
        bestIndices = [index for index in range(len(values)) if values[index] == maxValue]
        return legals[random.choice(bestIndices)]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
       """
    position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    consumedFoods = len(currentGameState.getFood().asList())
    ghosts = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghosts]
    evaluation = consumedFoods + currentGameState.getScore()
    length = len(foods)
    if length > 0:
        # calculate sum of distances to foods
        sumOfDistancesToFoods = 0
        for food in foods:
            sumOfDistancesToFoods += manhattanDistance(position, food)
        evaluation += (1.0 / sumOfDistancesToFoods)
    # calculate  sum of distances to ghosts
    sumOfDistancesToGhosts = 0
    for ghost in ghosts:
        sumOfDistancesToGhosts += manhattanDistance(position, ghost.getPosition())
    # calculate sum of scared times
    sumOfScaredTimes = sum(scaredTimes)
    if sumOfScaredTimes > 0:
        evaluation += sumOfScaredTimes - len(currentGameState.getCapsules()) - sumOfDistancesToGhosts
    else:
        evaluation += len(currentGameState.getCapsules()) + sumOfDistancesToGhosts
    return evaluation


# Abbreviation
better = betterEvaluationFunction
