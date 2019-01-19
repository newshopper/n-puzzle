"""
COMS W4701 Artificial Intelligence - Programming Homework 1

In this assignment you will implement and compare different search strategies
for solving the n-Puzzle, which is a generalization of the 8 and 15 puzzle to
squares of arbitrary size (we will only test it with 8-puzzles for now). 
See Courseworks for detailed instructions.

@author: BHASKAR GHOSH (bg2625)
"""

import time


def state_to_string(state):
    row_strings = [" ".join([str(cell) for cell in row]) for row in state]
    return "\n".join(row_strings)


def swap_cells(state, i1, j1, i2, j2):
    """
    Returns a new state with the cells (i1,j1) and (i2,j2) swapped. 
    """
    value1 = state[i1][j1]
    value2 = state[i2][j2]

    new_state = []
    for row in range(len(state)):
        new_row = []
        for column in range(len(state[row])):
            if row == i1 and column == j1:
                new_row.append(value2)
            elif row == i2 and column == j2:
                new_row.append(value1)
            else:
                new_row.append(state[row][column])
        new_state.append(tuple(new_row))
    return tuple(new_state)


# The following function returns the location of 0 in the given state.
#  It returns None if 0 is not found
def locate_zero(state):
    tupleNumber = 0
    zeroPositionInTuple = 0
    for targetTuple in state:
        if 0 in targetTuple:
            zeroPositionInTuple = targetTuple.index(0)
            if zeroPositionInTuple >= 0:
                return (tupleNumber, zeroPositionInTuple)
        tupleNumber += 1
    return None


def get_successors(state):
    """
    This function returns a list of possible successor states resulting
    from applicable actions. 
    The result should be a list containing (Action, state) tuples. 
    For example [("Up", ((1, 4, 2),(0, 5, 8),(3, 6, 7))), 
                 ("Left",((4, 0, 2),(1, 5, 8),(3, 6, 7)))] 
    """

    child_states = []
    zeroPosition = locate_zero(state)
    if zeroPosition != None:
        x = zeroPosition[0]
        y = zeroPosition[1]

        if y >= 0 and y < len(state) - 1:
            # create state for moving Left
            leftState = swap_cells(state, x, y, x, y + 1)
            child_states.append(("Left", leftState))

        if y > 0 and y < len(state):
            # create state for moving Right
            rightState = swap_cells(state, x, y, x, y - 1)
            child_states.append(("Right", rightState))

        if x >= 0 and x < len(state) - 1:
            # create state for moving up
            upState = swap_cells(state, x, y, x + 1, y)
            child_states.append(("Up", upState))

        if x > 0:
            # create state for moving Down
            downState = swap_cells(state, x, y, x - 1, y)
            child_states.append(("Down", downState))

    return child_states


def get_goal_state(state):
    """
    For given state, generate the goal state. It is easier to generate the goal state in the form of a list.
    """
    puzzle_order = len(state) ** 2
    # generate goal list
    goal_list = []
    for index in range(puzzle_order):
        goal_list.append(index)

    return goal_list

def goal_test(state):
    """
    Returns True if the state is a goal state, False otherwise. 
    """
    from itertools import chain

    # Get the goal state for given state
    goal_list = get_goal_state(state)
    # Convert state to list of individual elements
    state_list = list(chain(*state))
    if state_list == goal_list:
        return True
    else:
        return False


def bfs(state):
    """
    Breadth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a queue in BFS)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}
    action_list = []

    # get successors. From successors, extract the states and put them on to the fringe. Pop the parent state.
    # Add entry in dictionary with (parent state, child state) tuple and the action that was taken to traverse to the child state.
    currentState = state
    fringe.append(currentState)
    while goal_test(currentState) != True:
        if currentState in closed:
            # Run continue in this case. Pop the first element from the fringe
            if len(fringe) > 0:
                fringe.pop(0)
                currentState = fringe[0]
            continue
        else:
            closed.add(currentState)
        # get successors of givenState
        successors = get_successors(currentState)
        if len(successors) > 0:
            found_states_to_explore = False
            # iterate through all successors and add to parents dictionary
            for successor in successors:
                # Don't add child state (belonging to a successor) to the fringe or to parents dictionary
                # if it belongs to the closed set
                if successor[1] in closed:
                    continue
                # key, value is (parent state, child state), action
                parents[(currentState, successor[1])] = successor[0]
                fringe.append(successor[1])
                found_states_to_explore = True

            if found_states_to_explore == True:
                states_expanded += 1

            # Pop givenState from the fringe
            fringe.pop(0)
            if max_fringe < len(fringe):
                max_fringe = len(fringe)
            # Assign the first element of the fringe to givenState
            currentState = fringe[0]
        else:
            if len(fringe) > 0:
                fringe.pop(0)
                break

    else:
        # expanding goal state
        states_expanded += 1

    # trace parent of goal state using parents dictionary. Add corresponding action to actionList
    while currentState != state:
        for item in parents:
            if currentState == item[1]:
                currentState = item[0]
                action_list.insert(0, parents[item])
                break

    #  return solution, states_expanded, max_fringe
    return action_list, states_expanded, max_fringe

def dfs(state):
    """
    Depth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a stack in DFS)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}
    action_list = []

    # get successors. From successors, extract the states and put them on to the fringe. Pop the parent state.
    # Add entry in dictionary with (parent state, child state) tuple and the action that was taken to traverse to the child state.
    currentState = state
    fringe.append(currentState)
    while goal_test(currentState) != True:
        if currentState in closed:
            # Run continue in this case. Pop the first element from the fringe
            if len(fringe) > 0:
                fringe.pop()
                currentState = fringe[-1]
            continue

        # get successors of currentState
        successors = get_successors(currentState)
        if len(successors) > 0:
            # iterate through all successors and add to parents dictionary
            successorAdded = False
            for successor in successors:
                # Don't add child state (belonging to a successor) to the fringe or to parents dictionary
                # if it belongs to the closed set
                if successor[1] in closed:
                    continue

                # we do not want to add all child states to the fringe as we did in BFS. We have to choose only one.
                if successorAdded == False:
                    if successor[1] not in fringe:
                        # key, value is (parent state, child state), action
                        parents[(currentState, successor[1])] = successor[0]
                        # possible that there are states whose all nodes haven't been explored yet. One of the successor nodes
                        # will point to the parent of currentState. We don't want to add that to the fringe.
                        fringe.append(successor[1])
                        successorAdded = True
                        # we are ready to expand the state as it has an unexplored successor
                        states_expanded += 1


            if successorAdded == False:
                # this means no successor exists which is yet to be expanded
                closed.add(currentState)
                fringe.pop()

            if max_fringe < len(fringe):
                max_fringe = len(fringe)
            # Assign the last element of the fringe to currentState
            currentState = fringe[-1]
        else:
            closed.add(currentState)
            if len(fringe) > 0:
                fringe.pop()
                currentState = fringe[-1]
                break

    else:
        # expanding goal state
        states_expanded += 1

    # trace parent of goal state till the original state using parents dictionary. Add corresponding action to actionList
    while currentState != state:
        for item in parents:
            if currentState == item[1]:
                currentState = item[0]
                action_list.insert(0, parents[item])
                break

    #  return solution, states_expanded, max_fringe
    return action_list, states_expanded, max_fringe


def misplaced_heuristic(state):
    """
    Returns the number of misplaced tiles.
    """
    # iterate over all the tuples in the state
    rowNum = 0
    misplaced_tiles = 0
    for target_tuple in state:
        colNum = 0
        for item in target_tuple:
            if item == 0:
                colNum += 1
                continue
            if rowNum == 0:
                if item is not colNum:
                    misplaced_tiles += 1
            else:
                totalElementsBeforeCurrentElement = rowNum * len(target_tuple) + colNum
                if item is not totalElementsBeforeCurrentElement:
                    misplaced_tiles += 1
            colNum += 1
        rowNum += 1
    return misplaced_tiles


def manhattan_heuristic(state):
    """
    For each misplaced tile, compute the Manhattan distance between the current
    position and the goal position. Then return the sum of all distances.
    """

    # Convert the state to list representation
    state_list = []
    for target_tuple in state:
        target_list = list(target_tuple)
        state_list.extend(target_list)

    # create goal_list that basically represents the goal state
    goal_list = get_goal_state(state)

    # calculate difference between elements of both lists
    difference_list = []
    for i in range(len(goal_list)):
        # ignore those differences where an element is being compared to 0
        if state_list[i] != 0 and goal_list[i] != 0:
            difference_list.append(abs(state_list[i] - goal_list[i]))

    # difference_list = [abs(goalPosition - currentPosition) for goalPosition, currentPosition in zip(goal_list, state_list)]
    sum_difference = sum(difference_list)
    return sum_difference



def best_first(state, heuristic):
    """
    Best first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a priority queue in greedy search)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    # You may want to use these functions to maintain a priority queue
    from heapq import heappush
    from heapq import heappop

    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}
    action_list = []

    # get successors. From successors, extract the states and put them on to the fringe. Pop the parent state.
    # Add entry in dictionary with (parent state, child state) tuple and the action

    currentState = state
    heappush(fringe, (heuristic(currentState), currentState))

    while heuristic(currentState) != 0:
        if currentState in closed:
            # Run continue in this case. Pop the first element from the fringe
            if len(fringe) > 0:
                lowestCostNode = heappop(fringe)
                currentState = lowestCostNode[1]
            continue
        else:
            closed.add(currentState)
        # get successors of givenState
        successors = get_successors(currentState)
        if len(successors) > 0:
            found_states_to_explore = False
            # iterate through all successors and add to parents dictionary
            for successor in successors:
                # Don't add child state (belonging to a successor) to the fringe or to parents dictionary
                # if it belongs to the closed set
                if successor[1] in closed:
                    continue
                # key, value is (parent state, child state), action
                parents[(currentState, successor[1])] = successor[0]
                heappush(fringe, (heuristic(successor[1]), successor[1]))
                found_states_to_explore = True

            if found_states_to_explore == True:
                states_expanded += 1

            # Pop givenState from the fringe
            lowestCostNode = heappop(fringe)
            if max_fringe < len(fringe):
                max_fringe = len(fringe)
            # Assign the most recent popped element
            #  to currentState
            currentState = lowestCostNode[1]
        else:
            if len(fringe) > 0:
                lowestCostNode = heappop(fringe)
                currentState = lowestCostNode[1]
                break

    else:
        # expanding goal state
        states_expanded += 1

    # trace parent of goal state using parents dictionary. Add corresponding action to actionList
    while currentState != state:
        for item in parents:
            if currentState == item[1]:
                currentState = item[0]
                action_list.insert(0, parents[item])
                break

    #  return solution, states_expanded, max_fringe
    return action_list, states_expanded, max_fringe


def astar(state, heuristic):
    """
    A-star search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a priority queue in greedy search)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    # You may want to use these functions to maintain a priority queue
    from heapq import heappush
    from heapq import heappop

    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}
    costs = {}
    action_list = []

    currentState = state
    # for the initial state, background cost would be 0
    costs[currentState] = 0
    heappush(fringe, (heuristic(currentState), currentState))

    while heuristic(currentState) != 0:
        if currentState in closed:
            # Run continue in this case. Pop the first element from the fringe
            if len(fringe) > 0:
                lowestCostNode = heappop(fringe)
                currentState = lowestCostNode[1]
            continue
        else:
            closed.add(currentState)
        # get successors of givenState
        successors = get_successors(currentState)
        if len(successors) > 0:
            found_states_to_explore = False
            # iterate through all successors and add to parents dictionary
            for successor in successors:
                # Don't add child state (belonging to a successor) to the fringe or to parents dictionary
                # if it belongs to the closed set
                if successor[1] in closed:
                    continue
                # key, value is (parent state, child state), action
                parents[(currentState, successor[1])] = successor[0]
                # calculate cost for successor node based on cost of parent node
                cost_parent = costs[currentState]
                cost_successor = cost_parent + 1
                costs[successor[1]] = cost_successor
                # Push successor onto the fringe
                heappush(fringe, (cost_successor + heuristic(successor[1]), successor[1]))
                found_states_to_explore = True

            if found_states_to_explore == True:
                states_expanded += 1

            # Pop givenState from the fringe
            lowestCostNode = heappop(fringe)
            if max_fringe < len(fringe):
                max_fringe = len(fringe)
            # Assign the most recent popped element
            #  to currentState
            currentState = lowestCostNode[1]
        else:
            if len(fringe) > 0:
                lowestCostNode = heappop(fringe)
                currentState = lowestCostNode[1]
                break

    else:
        # expanding goal state
        states_expanded += 1

    # trace parent of goal state using parents dictionary. Add corresponding action to actionList
    while currentState != state:
        for item in parents:
            if currentState == item[1]:
                currentState = item[0]
                action_list.insert(0, parents[item])
                break

    #  return solution, states_expanded, max_fringe
    return action_list, states_expanded, max_fringe


def print_result(solution, states_expanded, max_fringe):
    """
    Helper function to format test output. 
    """
    if solution is None:
        print("No solution found.")
    else:
        print("Solution has {} actions.".format(len(solution)))
    print("Total states expanded: {}.".format(states_expanded))
    print("Max fringe size: {}.".format(max_fringe))


if __name__ == "__main__":

    # Easy test case
    test_state = ((1, 4, 2),
                  (0, 5, 8),
                  (3, 6, 7))

    goal_state = ((0, 1, 2),
                  (3, 4, 5),
                  (6, 7, 8))

    # More difficult test case
    hard_test_state = ((7, 2, 4),
                       (5, 0, 6),
                       (8, 3, 1))

    isGoalState = goal_test(hard_test_state)
    print(isGoalState)

    print(get_successors(test_state))

    print(state_to_string(hard_test_state))
    print()

    print(misplaced_heuristic(test_state))
    print(manhattan_heuristic(hard_test_state))

    """
    bfs_state = ((1, 2, 0),
                 (3, 4, 5),
                 (6, 7, 8))

    print("====BFS====")
    start = time.time()
    solution, states_expanded, max_fringe = bfs(test_state)  #
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
      print(solution)
    print("Total time: {0:.3f}s".format(end - start))

    print()
    print("====DFS====")
    start = time.time()
    solution, states_expanded, max_fringe = dfs(test_state)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
       print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    

    print()
    print("====Greedy Best-First (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_fringe = best_first(hard_test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====A* (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_fringe = astar(hard_test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====A* (Total Manhattan Distance Heuristic)====")
    start = time.time()
    solution, states_expanded, max_fringe = astar(hard_test_state, manhattan_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))
    
    """
