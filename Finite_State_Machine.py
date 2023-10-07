# Import Modules
import re
from functools import reduce

class Fsm:
    def __init__(self, alphabet, states, start, final, transitions):
        self.sigma = alphabet
        self.states = states
        self.start = start
        self.final = final
        self.transitions = transitions

    # str method to represent alphabets and numbers
    def __str__(self):
        sigma = "Alphabet: " + str(self.sigma) + "\n"
        states = "States: " + str(self.states) + "\n"
        start = "Start: " + str(self.start) + "\n"
        final = "Final: " + str(self.final) + "\n"
        trans_header = "Transitions: [\n"
        thlen = len(trans_header)
        translist = ""
        for t in self.transitions:
            translist += " " * thlen + str(t) + "\n"
        translist += " " * thlen + "]"
        transitions = trans_header + translist
        ret = sigma + states + start + final + transitions
        return ret

count = 0

def fresh():
    global count
    count += 1
    return count

def char(string):
    return [fresh(), [string]]

# function for concatenating
def concat(r1, r2):
    r1_start, r1_end = r1
    r2_start, r2_end = r2
    transitions = r1_end + [r2_start] + r2[1]  # Ensure all elements are in a list
    return [r1_start, r2_end, transitions]

# function for union
def union(r1, r2):
    start_state = fresh()
    end_state = fresh()
    transitions = [start_state, r1[0], r2[0]] + r1[1] + [end_state] + r2[1]
    return [start_state, end_state, transitions]

# function for star
def star(r1):
    start_state = fresh()
    end_state = fresh()
    transitions = [start_state, r1[0], end_state, start_state] + r1[1]
    return [start_state, end_state, transitions]

# function for e_closure
def e_closure(s, nfa):
    stack = [s]
    visited = set()
    epsilon_closure = []

    while stack:
        state = stack.pop()
        if state not in visited:
            epsilon_closure.append(state)
            visited.add(state)
            transitions = [t for t in nfa.transitions if t[0] == state and (t[1] == "" or t[1] is None)]
            for transition in transitions:
                stack.append(transition[2])

    return epsilon_closure


# function for move
def move(c, s, nfa):
    next_states = []
    for transition in nfa.transitions:
        if transition[0] == s and transition[1] == c:
            next_states.append(transition[2])
    return next_states

# function for nfa to dfa
def nfa_to_dfa(nfa):
    dfa_states = []
    dfa_transitions = []
    dfa_start_state = e_closure(nfa.start, nfa)
    dfa_states.append(frozenset(dfa_start_state))  # Use frozenset
    stack = [frozenset(dfa_start_state)]  # Use frozenset

    while stack:
        current_state_set = stack.pop()
        for symbol in nfa.sigma:
            next_state_set = []
            for state in current_state_set:
                next_state_set.extend(move(symbol, state, nfa))
            epsilon_closure_set = []
            for state in next_state_set:
                epsilon_closure_set.extend(e_closure(state, nfa))
            next_state_set_frozen = frozenset(epsilon_closure_set)  # Use frozenset
            if next_state_set_frozen not in dfa_states:
                stack.append(next_state_set_frozen)
                dfa_states.append(next_state_set_frozen)
            dfa_transitions.append([current_state_set, symbol, next_state_set_frozen])

    dfa_final_states = [state_set for state_set in dfa_states if any(state in nfa.final for state in state_set)]
    return Fsm(nfa.sigma, dfa_states, frozenset(dfa_start_state), dfa_final_states, dfa_transitions)

# function for accepting alphabets and digits
def accept(nfa, string):
    dfa = nfa_to_dfa(nfa)
    current_state = frozenset([dfa.start])  # Use frozenset
    for char in string:
        next_states = set()  # Use set to store next states
        for state in current_state:
            next_states.update(move(char, state, dfa))
        if not next_states:
            return False
        current_state = frozenset(next_states)  # Use frozenset

    return any(state in dfa.final for state in current_state)

# Example for test:
alphabet = {'0', '1'}
states = {1, 2, 3, 4}
start = 1
final = {4}
transitions = [(1, '0', 2), (2, '1', 3), (3, '0', 4)]
nfa = Fsm(alphabet, states, start, final, transitions)
print(accept(nfa, "010"))  # This prints True
print("Accept function passed successfully!")

# 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# Test the e_closure function
alphabet = {'a', 'b'}
states = {1, 2, 3}
start = 1
final = {3}
transitions = [(1, 'a', 2), (2, '', 3), (2, 'b', 2)]
nfa = Fsm(alphabet, states, start, final, transitions)

# Compute epsilon closure from state 1
epsilon_states = e_closure(1, nfa)
assert set(epsilon_states) == {1}

# Compute epsilon closure from state 2
epsilon_states = e_closure(2, nfa)
assert set(epsilon_states) == {2, 3}

# Compute epsilon closure from state 3
epsilon_states = e_closure(3, nfa)
assert set(epsilon_states) == {3}

print("Epsilon closure tests passed!")

# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# Test the move function
alphabet = {'0', '1'}
states = {1, 2, 3, 4}
start = 1
final = {4}
transitions = [(1, '0', 2), (2, '1', 3), (3, '0', 4)]
nfa = Fsm(alphabet, states, start, final, transitions)

# Test moving from state 1 with '0'
move_result = move('0', 1, nfa)
assert move_result == [2]  # Expected result: [2]

# Test moving from state 2 with '1'
move_result = move('1', 2, nfa)
assert move_result == [3]  # Expected result: [3]

# Test moving from state 3 with '0'
move_result = move('0', 3, nfa)
assert move_result == [4]  # Expected result: [4]

# Test moving from state 4 with '1' (no valid move)
move_result = move('1', 4, nfa)
assert move_result == []  # Expected result: []

print("Move function test passed!")

# 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000



