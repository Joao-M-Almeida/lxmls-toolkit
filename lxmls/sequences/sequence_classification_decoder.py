import numpy as np
from lxmls.sequences.log_domain import *
import pdb


class SequenceClassificationDecoder:
    """ Implements a sequence classification decoder."""

    def __init__(self):
        pass

    # ----------
    # Computes the forward trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_forward(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Forward variables.
        forward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        forward[0, :] = emission_scores[0, :] + initial_scores

        # Forward loop.
        for pos in xrange(1, length):
            for current_state in xrange(num_states):
                # Note the fact that multiplication in log domain turns a sum and sum turns a logsum
                forward[pos, current_state] = logsum(forward[pos-1, :] + transition_scores[pos-1, current_state, :])
                forward[pos, current_state] += emission_scores[pos, current_state]

        # Termination.
        log_likelihood = logsum(forward[length-1, :] + final_scores)

        return log_likelihood, forward

    # ----------
    # Computes the backward trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_backward(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Backward variables.
        backward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        backward[length-1, :] = final_scores

        # Backward loop.
        for pos in xrange(length-2, -1, -1):
            for current_state in xrange(num_states):
                backward[pos, current_state] = \
                    logsum(backward[pos+1, :] +
                           transition_scores[pos, :, current_state] +
                           emission_scores[pos+1, :])

        # Termination.
        log_likelihood = logsum(backward[0, :] + initial_scores + emission_scores[0, :])

        return log_likelihood, backward

    # ----------
    # Computes the viterbi trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_viterbi(self, initial_scores, transition_scores, final_scores, emission_scores):

        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, num_states]) + logzero()

        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, num_states], dtype=int)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Complete Exercise 2.8
        #raise NotImplementedError("Complete Exercise 2.8")

        #### Little guide of the implementation ####################################
        # Initializatize the viterbi scores
        #
        # Do the double of the viterbi loop (lines 7 to 12 in the guide pdf)
        # from 1 to length
        #     from 0 to num_states
        #       ...
        #
        # define the best_path and best_score
        #
        # backtrack the best_path using the viterbi paths (lines 17-18 pseudocode in the guide pdf)
        #
        # return best_path and best_score
        ############################################################################

        for i in xrange(num_states):
            #viterbi_scores[0, i] = logsum_pair(initial_scores[i], emission_scores[0, i])
            viterbi_scores[0, i] = initial_scores[i] + emission_scores[0, i]

        for i in xrange(1, length):
            for k in xrange(num_states):
                viterbi_scores[i, k] = (np.max(transition_scores[i-1, k, :] + viterbi_scores[i-1, :]) + emission_scores[i, k])
                #import pdb; pdb.set_trace()
                viterbi_paths[i, k] = np.argmax(transition_scores[i-1, k, :] + viterbi_scores[i-1, :])

        best_score = np.max(final_scores + viterbi_scores[length-1])

        best_path[length-1] = np.argmax(final_scores + viterbi_scores[length-1])



        for i in reversed(xrange(0, length-1)):
            best_path[i] = viterbi_paths[i+1, best_path[i+1]]

        return best_path, best_score

    def run_forward_backward(self, initial_scores, transition_scores, final_scores, emission_scores):
        log_likelihood, forward = self.run_forward(initial_scores, transition_scores, final_scores, emission_scores)
        print 'Log-Likelihood =', log_likelihood

        log_likelihood, backward = self.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
        print 'Log-Likelihood =', log_likelihood

        return forward, backward
