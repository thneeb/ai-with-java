package de.neebs.ai.control.perceptron;

import java.util.Random;

public class MachineLearning {
    private double[] inputs;
    private Neuron neuron;

    public MachineLearning() {
        int numberOfInputSignals = TrainingParameter.inputs[0].length;
        inputs = new double[numberOfInputSignals];
        neuron = new Neuron(numberOfInputSignals, null);
        ProcessMonitoring.lastWeights = neuron.getWeights();
    }
    
    public void showWeights() {
        System.out.print("weights incl. bias:  ");
        for (int i = 0; i < ProcessMonitoring.lastWeights.length; i++) {
            System.out.print(ProcessMonitoring.lastWeights[i] + " ");
        }
        System.out.println();
    }

    public void trainWithSupervisedLearning() {
        Random random = new Random();
        for (int epoche = 1; epoche <= TrainingParameter.numberOfEpochs; epoche++) {
            int sample = random.nextInt(TrainingParameter.inputs.length);
            System.out.println("epoche: " + epoche + "   data number: " + (sample + 1));
            calculateOutput(TrainingParameter.inputs[sample],TrainingParameter.activationFunction);
            if (TrainingParameter.targets[sample] - ProcessMonitoring.lastOutputWithActivationFunction != 0) {
                calculateNewWeights(TrainingParameter.targets[sample]);
            }
        }
        System.out.println("end of training\n");
    }

    public void calculateOutput(double[] inputs, ActivationFunction activationFunction) {
        this.inputs = inputs;
        ProcessMonitoring.lastOutputAsSum = neuron.getOutput(inputs, ActivationFunction.ONLYSUM);
        ProcessMonitoring.lastOutputWithActivationFunction =
                neuron.getOutput(inputs, activationFunction);
        ProcessMonitoring.lastWeights = neuron.getWeights();
    }

    private void calculateNewWeights(double target) {
        double error = target - ProcessMonitoring.lastOutputWithActivationFunction;
        System.out.print("error: " + error + "\nweight adjustment" + "\nnew weights:  ");
        for (int i = 0; i < ProcessMonitoring.lastWeights.length; i++) {
            double newWeight = neuron.getWeights(i) + calculateDeltaW(i, error);
            neuron.setWeights(i, newWeight);
            System.out.print(newWeight + "  ");
        }
        System.out.println();
    }

    private double calculateDeltaW(int i, double error) {
        double deltaW = error * TrainingParameter.learningRate;   // delta rule
        deltaW *= (i < inputs.length) ? inputs[i] : 1;            // weight : bias
        return deltaW;
    }

    public void testAllInputsAndShowResults() {
        int learnedInPercent = 0;
        System.out.print("outputs: ");
        for (int i = 0; i < TrainingParameter.inputs.length; i++) {
            calculateOutput(TrainingParameter.inputs[i], TrainingParameter.activationFunction);
            System.out.print(ProcessMonitoring.lastOutputAsSum + "  ");
            if (ProcessMonitoring.lastOutputWithActivationFunction == TrainingParameter.targets[i]) {
                learnedInPercent++;
            }
        }
        learnedInPercent /= TrainingParameter.inputs.length * 0.01;
        System.out.println("\nlearned in percent: " + learnedInPercent + "\n");
    }
}
