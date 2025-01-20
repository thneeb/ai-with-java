package de.neebs.ai.control.rl;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public interface NeuralNetworkFactory {
    MultiLayerNetwork createNeuralNetwork();
}
