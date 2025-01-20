package de.neebs.ai.control.perceptron;

import org.springframework.stereotype.Service;

@Service
public class PerceptronMain {
    public void execute() {
        MachineLearning machineLearning = new MachineLearning();
        machineLearning.showWeights();
        machineLearning.testAllInputsAndShowResults();
        machineLearning.trainWithSupervisedLearning();
        machineLearning.showWeights();
        machineLearning.testAllInputsAndShowResults();
    }
}
