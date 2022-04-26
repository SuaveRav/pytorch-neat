import torch
import logging
import torch.nn as nn
import pickle
import numpy as np
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

    ACTIVATION = 'relu'
    SCALE_ACTIVATION = 1

    FITNESS_THRESHOLD = 1950

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 35
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.2
    ADD_CONNECTION_MUTATION_RATE = 0.6

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    PERCENTAGE_TO_SAVE = 0.3  # Top percentage of species to be saved before mating

    # Load Data
    print("Loading chess data")
    print(DEVICE)
    
    with open("./data/KRK/indices", "rb") as f: 
        data = pickle.load(f)

    inputs_list = []
    outputs_list = []

    np_rng.shuffle(data)
    data = data[:2000]

    for d in data:
        d_in = np.array(d[0]) / 8
        # condensed_input = np.argmax(d_in.reshape(4, 64), axis=1)
        inputs_list.append(d_in)
        outputs_list.append(d[1])

    inputs = list(map(lambda s: autograd.Variable(torch.Tensor([s])), inputs_list))
    targets = list(map(lambda s: autograd.Variable(torch.Tensor([s])), outputs_list))

    def fitness_fn(self, genome):
        fitness = 2000  # Max fitness

        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        criterion = nn.MSELoss()
        num_inputs = len(self.inputs)
        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            pred = phenotype(input)

            loss = (float(pred) - float(target)) ** 2
            loss = float(loss)
            # loss = criterion(pred, target)
            # logger.info("Loss: {}".format(loss))
            fitness -= loss
            # logger.info("Fitness: {}".format(fitness))
        # fitness = fitness / num_inputs
        # logger.info("Fitness: {}".format(fitness))
        return fitness

    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))

        return predictions, labels
