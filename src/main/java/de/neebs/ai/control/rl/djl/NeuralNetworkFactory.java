package de.neebs.ai.control.rl.djl;

import ai.djl.Model;
import ai.djl.training.Trainer;

public interface NeuralNetworkFactory {
    Model createNeuralNetwork(long seed);

    Trainer createTrainer(Model model);
}
