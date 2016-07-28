from __future__ import division
import sys
import numpy as np
import lxmls.sequences.discriminative_sequence_classifier as dsc
import pdb


class StructuredPerceptron(dsc.DiscriminativeSequenceClassifier):
    """ Implements a first order CRF"""

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 num_epochs=10, learning_rate=1.0, averaged=True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []

    def train_supervised(self, dataset):
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        num_examples = dataset.size()
        for epoch in xrange(self.num_epochs):
            num_labels_total = 0
            num_mistakes_total = 0
            for i in xrange(num_examples):
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
            self.params_per_epoch.append(self.parameters.copy())
            acc = 1.0 - num_mistakes_total / num_labels_total
            print "Epoch: %i Accuracy: %f" % (epoch, acc)
        self.trained = True

        if self.averaged:
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w /= len(self.params_per_epoch)
            self.parameters = new_w

    def perceptron_update(self, sequence):
        num_labels = len(sequence.x)
        num_mistakes = 0
        # predict sequence to get ^y

        # for each ^y_i:
        #   if y_i != ^y_i:
        #       get features for y_i
        #       add self.learning_rate to features that fired for y
        #       get features for ^y
        #       subtract self.learning_rate from features that fired for y^


        pred_seq, _ = self.viterbi_decode(sequence)
        y_pred = pred_seq.y
        y_t_true = sequence.y

        # Update initial features.
        if y_pred[0] !=  y_t_true[0]:
            true_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_true[0])
            self.parameters[true_initial_features] += self.learning_rate
            pred_initial_features = self.feature_mapper.get_initial_features(sequence, y_pred[0])
            self.parameters[pred_initial_features] -= self.learning_rate

        for pos in xrange(num_labels):
            if y_t_true[pos] != y_pred[pos]:

                num_mistakes += 1

                # Update emission features.
                true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true[pos])
                self.parameters[true_emission_features] += self.learning_rate
                pred_emission_features = self.feature_mapper.get_emission_features(
                    sequence, pos, y_pred[pos])
                self.parameters[pred_emission_features] -= self.learning_rate

            if pos > 0:
                if y_t_true[pos] != y_pred[pos] or y_t_true[pos-1] != y_pred[pos-1]:
                    # Update transition features.
                    prev_y_t_true = y_t_true[pos-1]
                    true_transition_features = self.feature_mapper.get_transition_features(
                        sequence, pos-1, y_t_true[pos], prev_y_t_true)
                    self.parameters[true_transition_features] += self.learning_rate
                    pred_transition_features = self.feature_mapper.get_transition_features(
                        sequence, pos-1, y_pred[pos], y_pred[pos-1])
                    self.parameters[pred_transition_features] -= self.learning_rate

        # Update Final features.
        if y_pred[num_labels-1] !=  y_t_true[num_labels-1]:
            true_final_features = self.feature_mapper.get_final_features(
                sequence, y_t_true[num_labels-1])
            self.parameters[true_final_features] += self.learning_rate
            pred_final_features = self.feature_mapper.get_final_features(
                sequence, y_pred[num_labels-1])
            self.parameters[pred_final_features] -= self.learning_rate

        return (num_labels, num_mistakes)

    def save_model(self, dir):
        fn = open(dir + "parameters.txt", 'w')
        for p_id, p in enumerate(self.parameters):
            fn.write("%i\t%f\n" % (p_id, p))
        fn.close()

    def load_model(self, dir):
        fn = open(dir + "parameters.txt", 'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
