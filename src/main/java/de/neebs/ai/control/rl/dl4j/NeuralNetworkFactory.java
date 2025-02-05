package de.neebs.ai.control.rl.dl4j;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public interface NeuralNetworkFactory {
    MultiLayerNetwork createNeuralNetwork(long seed);
}
