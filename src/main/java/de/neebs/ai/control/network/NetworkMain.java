package de.neebs.ai.control.network;

import org.springframework.stereotype.Service;

@Service
public class NetworkMain {
    public void execute() {
        FeedForwardNetwork neuronalNetwork = new FeedForwardNetwork(
                NetworkParameter.numberOfInputSignals,
                NetworkParameter.numberOfNeuronsInHiddenLayer,
                NetworkParameter.numberOfNeuronsInOutputLayer);
        DisplayMachineLearning.showWeights(neuronalNetwork.getWeightsOfHiddenLayer(),
                neuronalNetwork.getWeightsOfOutputLayer());
        neuronalNetwork.testAllInputsAndShowResults();
        neuronalNetwork.trainWithSupervisedLearning();
        DisplayMachineLearning.showWeights(neuronalNetwork.getWeightsOfHiddenLayer(),
                neuronalNetwork.getWeightsOfOutputLayer());
        neuronalNetwork.testAllInputsAndShowResults();
    }
}
