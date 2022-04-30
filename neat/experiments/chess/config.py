import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import autograd

from neat.phenotype.feed_forward import FeedForwardNet


logger = logging.getLogger(__name__)

class ChessConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np_rng = np.random.default_rng(13)

    # DEVICE = "cuda:0"
    VERBOSE = True

    NUM_INPUTS = 8
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'wann'
    SCALE_ACTIVATION = 1

    FITNESS_THRESHOLD = 1950

    POPULATION_SIZE = 250
    NUMBER_OF_GENERATIONS = 2
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ACTIVATION_MUTATION_RATE = 0.2
    ADD_NODE_MUTATION_RATE = 0.1
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    PERCENTAGE_TO_SAVE = 0.3  # Top percentage of species to be saved before mating

    NORMALIZE_INPUTS = True

    # Load Data
    print("Loading chess data")
    print(DEVICE)
    
    with open("./data/KRK/indices", "rb") as f: 
        data = pickle.load(f)

    inputs_list = []
    outputs_list = []

    np_rng.shuffle(data)

    data_proportion = 2000
    data_train = data[:data_proportion]

    for d in data_train:
        if NORMALIZE_INPUTS:
            d_in = np.array(d[0] / 8)
        else:
            d_in = np.array(d[0])
        # condensed_input = np.argmax(d_in.reshape(4, 64), axis=1)
        inputs_list.append(d_in)
        outputs_list.append(d[1])

    inputs = torch.tensor(np.array(inputs_list)).to(DEVICE)
    targets = torch.tensor(np.array(outputs_list)).reshape(-1, 1).to(DEVICE)
    
    # Testing Data
    inputs_list_test = []
    outputs_list_test = []

    data_proportion_test = 2000
    data_proportion_test = data_proportion + data_proportion_test
    data_test = data[data_proportion:data_proportion_test]

    for d in data_test:
        if NORMALIZE_INPUTS:
            d_in = np.array(d[0] / 8)
        else:
            d_in = np.array(d[0])
        # condensed_input = np.argmax(d_in.reshape(4, 64), axis=1)
        inputs_list_test.append(d_in)
        outputs_list_test.append(d[1])

    inputs_test = torch.tensor(np.array(inputs_list_test)).to(DEVICE)
    targets_test = torch.tensor(np.array(outputs_list_test)).reshape(-1, 1).to(DEVICE)
        
    def fitness_fn(self, genome):
        fitness = self.data_proportion  # Max fitness

        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        criterion = nn.MSELoss()
        num_inputs = len(self.inputs)

        pred = phenotype(self.inputs)
        loss = criterion(pred, self.targets)

        fitness -= self.data_proportion * loss.item()

        return fitness

    def test(self, best_network):
        best_network.to(self.DEVICE)
        criterion = nn.MSELoss()
        predictions = []
        losses = []

        for i in range(len(self.inputs_test)):
            input = self.inputs_test[i]
            target = self.targets_test[i]
            pred = best_network(torch.reshape(input,(1,8)))
            loss = criterion(pred[0], target)
            losses.append([loss.detach().cpu().numpy()])
            predictions.append(pred[0].detach().cpu().numpy())

        results = self.inputs_test.detach().cpu().numpy()
        results = np.hstack((results, losses))
        results = np.hstack((results, predictions))
        return results


    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets):
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))

        return predictions, labels
