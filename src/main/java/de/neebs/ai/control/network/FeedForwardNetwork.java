package de.neebs.ai.control.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FeedForwardNetwork {
    private double[] inputLayer;
    private List<Neuron> hiddenLayer;
    private List<Neuron> outputLayer;

    public FeedForwardNetwork(int numberOfNetworkInputSignals,
                              int numberOfNeuronsInHiddenLayer,
                              int numberOfNeuronsInOutputLayer) {
        inputLayer = new double[numberOfNetworkInputSignals];
        builtHiddenLayer(inputLayer.length, numberOfNeuronsInHiddenLayer);
        builtOutputLayer(hiddenLayer.size(), numberOfNeuronsInOutputLayer);
        ProcessMonitoring.lastOutputs = new double[numberOfNeuronsInOutputLayer];
        ProcessMonitoring.lastOutputsFromHiddenLayer = new double[numberOfNeuronsInHiddenLayer];
        if (TrainingParameter.weightsOfHiddenLayer != null)
            setWeightsOfHiddenLayer();  // in the case of given weights (hidden layer)
        if (TrainingParameter.weightsOfOutputLayer != null)
            setWeightsOfOutputLayer();  // in the case of given weights (output layer)
    }

    private void builtHiddenLayer(int numberOfLayerInputSignals,
                                  int numberOfNeuronsInHiddenLayer) {
        hiddenLayer = new ArrayList<>();
        double bias = 2 * new Random().nextDouble() - 1;
        for (int i = 0; i < numberOfNeuronsInHiddenLayer; i++)
            hiddenLayer.add(new Neuron(numberOfLayerInputSignals, outputLayer, bias));
    }

    private void builtOutputLayer(int numberOfLayerInputSignals,
                                  int numberOfNeuronsInOutputLayer) {
        outputLayer = new ArrayList<>();
        double bias = 2 * new Random().nextDouble() - 1;
        for (int i = 0; i < numberOfNeuronsInOutputLayer; i++)
            outputLayer.add(new Neuron(numberOfLayerInputSignals, null, bias));
    }

    public void setWeightsOfHiddenLayer() {
        for (int i = 0; i < hiddenLayer.size(); i++)
            for (int j = 0; j < inputLayer.length + 1; j++)            // + 1 (bias)
                hiddenLayer.get(i).setWeights(j, TrainingParameter.weightsOfHiddenLayer[i][j]);
    }

    public double[][] getWeightsOfHiddenLayer() {
        double[][] weightsOfHiddenLayer =
                new double[hiddenLayer.size()][hiddenLayer.size() + 1];    // + 1 (bias)
        for (int i = 0; i < hiddenLayer.size(); i++)
            for (int j = 0; j < inputLayer.length + 1; j++)            // + 1 (bias)
                weightsOfHiddenLayer[i][j] = hiddenLayer.get(i).getWeights()[j];
        return weightsOfHiddenLayer;
    }

    public void setWeightsOfOutputLayer() {
        for (int i = 0; i < outputLayer.size(); i++)
            for (int j = 0; j < hiddenLayer.size() + 1; j++)           // + 1 (bias)
                outputLayer.get(i).setWeights(j, TrainingParameter.weightsOfOutputLayer[i][j]);
    }

    public double[][] getWeightsOfOutputLayer() {
        double[][] weightsOfOutputLayer =
                new double[outputLayer.size()][hiddenLayer.size() + 1];    // + 1 (bias)
        for (int i = 0; i < outputLayer.size(); i++)
            for (int j = 0; j < hiddenLayer.size() + 1; j++)           // + 1 (bias)
                weightsOfOutputLayer[i][j] = outputLayer.get(i).getWeights()[j];
        return weightsOfOutputLayer;
    }

    public void calculateOutput(double[] input) {
        setInputLayer(input);
        calculateOutputFromHiddenLayer(input);
        calculateOutputFromOutputLayer();
    }

    private void setInputLayer (double[] inputLayer) {
        this.inputLayer = inputLayer;
    }

    private void calculateOutputFromHiddenLayer(double[] inputOfInputLayer) {
        for (int i = 0; i < hiddenLayer.size(); i++)
            ProcessMonitoring.lastOutputsFromHiddenLayer[i] = hiddenLayer.get(i).getOutput(
                    inputOfInputLayer, TrainingParameter.activationFunction);
    }

    private void calculateOutputFromOutputLayer() {
        for (int i = 0; i < outputLayer.size(); i++)
            ProcessMonitoring.lastOutputs[i] = outputLayer.get(i).getOutput(
                    ProcessMonitoring.lastOutputsFromHiddenLayer, TrainingParameter.activationFunction);
    }

    public void trainWithSupervisedLearning() {
        for (int epoche = 1; epoche <= TrainingParameter.numberOfEpochs; epoche++) {
            int sample = new Random().nextInt(TrainingParameter.inputs.length);
            System.out.println("\nsample: " + sample + "   epoche: " +
                    epoche + "   number of data: " + (sample + 1));
            calculateOutput(TrainingParameter.inputs[sample]);
            DisplayMachineLearning.showOutput(ProcessMonitoring.lastOutputs);
            double totalError = calculateTotalError(TrainingParameter.targets[sample]);
            System.out.println("\ntotal error before: " + totalError);
            makeBackPropagationForOutputLayer(TrainingParameter.targets[sample]);
            makeBackPropagationForHiddenLayer(TrainingParameter.inputs[sample],
                    TrainingParameter.targets[sample]);
            calculateOutput(TrainingParameter.inputs[sample]);
            totalError = calculateTotalError(TrainingParameter.targets[sample]);
            System.out.println("\ntotal error after backpropagation: " + totalError);
        }
    }

    private double calculateTotalError(double[] targets) {
        double totalError = 0;
        for (int i = 0; i < ProcessMonitoring.lastOutputs.length; i++)
            totalError += 0.5 * Math.pow(ProcessMonitoring.lastOutputs[i]- targets[i], 2);
        return totalError;
    }

    public void testAllInputsAndShowResults() {
        int learnedInPercent = 0;
        double totalErrorAtAll = 0;
        System.out.println();
        for (int i = 0; i < TrainingParameter.inputs.length; i++) {
            calculateOutput(TrainingParameter.inputs[i]);
            System.out.print("output " + (i+1) + ": ");
            totalErrorAtAll += calculateTotalError(TrainingParameter.targets[i]);
            boolean areAllOutputsOK = true;
            for (int j = 0; j < TrainingParameter.targets[0].length; j++) {
                System.out.print(ProcessMonitoring.lastOutputs[j] + "  ");
                if (Math.abs(ProcessMonitoring.lastOutputs[j] - TrainingParameter.targets[i][j]) >
                        TrainingParameter.faultTolerance)
                    areAllOutputsOK = false;
            }
            if (areAllOutputsOK)
                learnedInPercent++;
            System.out.println();
        }
        learnedInPercent /= TrainingParameter.inputs.length * 0.01;
        totalErrorAtAll /= TrainingParameter.inputs.length;
        System.out.println("\naverage of total errors: " + totalErrorAtAll);
        System.out.println("\nlearned in percent: " + learnedInPercent);
    }

    private void makeBackPropagationForOutputLayer(double[] targets) {
        System.out.println("\nbackpropagation output layer");
        double[] possibleBiasValuesForOutputlayer = new double[targets.length];
        for (int i = 0; i < ProcessMonitoring.lastOutputs.length; i++) {
            double lastOutputI = ProcessMonitoring.lastOutputs[i];
            System.out.println("\noutput: " + lastOutputI + "     target: " + targets[i] + "\n");
            int numberOfWeightsInclBias = outputLayer.get(i).getWeights().length;
            for (int j = 0; j < numberOfWeightsInclBias; j++) {
                double singleWeightError = (targets[i] - lastOutputI) * (lastOutputI) *
                        (1 - lastOutputI);
                double deltaW = singleWeightError * TrainingParameter.learningRate;
                deltaW *= (isNotABiasWeight(j, numberOfWeightsInclBias)) ?
                        ProcessMonitoring.lastOutputsFromHiddenLayer[j] : 1;
                double newWeight = outputLayer.get(i).getWeights(j) + deltaW;
                if (isNotABiasWeight(j, numberOfWeightsInclBias)) {
                    outputLayer.get(i).setWeights(j, newWeight);
                    System.out.println("new weight: " + newWeight);
                }
                else {
                    System.out.println("bias calculation " + (i + 1) + ": " + newWeight);
                    possibleBiasValuesForOutputlayer[i] = newWeight;
                }
            }
        }
        if (TrainingParameter.isBiasBackPropagationDesired)
            setBias(possibleBiasValuesForOutputlayer, outputLayer);
    }

    private boolean isNotABiasWeight(int j, int numberOfWeightsInclBias) {
        return j < numberOfWeightsInclBias - 1;
    }

    private void setBias(double[] bias, List<Neuron> layer) {
        double average = 0;
        for (double possibleValue : bias)
            average += possibleValue;
        average /= bias.length;
        for (int i = 0; i < layer.size(); i++)
            layer.get(i).setWeights(layer.get(i).getWeights().length - 1, average);
    }

    private void makeBackPropagationForHiddenLayer(double[] trainingData, double[] targets) {
        System.out.println("\nbackpropagation hidden layer\n");
        double[] possibleBiasValuesForHiddenLayer = new double[hiddenLayer.size()];
        for (int i = 0; i < hiddenLayer.size(); i++) {
            int numberOfWeightsInclBias = hiddenLayer.get(i).getWeights().length;
            for (int j = 0; j < numberOfWeightsInclBias; j++) {
                double deltaW = calculateDeltaW(i, j, trainingData, targets);
                double newWeight = hiddenLayer.get(i).getWeights(j) + deltaW;
                if (isNotABiasWeight(j, numberOfWeightsInclBias)) {
                    System.out.println("weight adjustment: " + deltaW + "\nnew weight: " + newWeight);
                    hiddenLayer.get(i).setWeights(j, newWeight);
                }
                else {
                    System.out.println("bias calculation " + (i + 1) + ": " + newWeight);
                    possibleBiasValuesForHiddenLayer[i] = newWeight;
                }
            }
        }
        if (TrainingParameter.isBiasBackPropagationDesired)
            setBias(possibleBiasValuesForHiddenLayer, hiddenLayer);
    }

    private double calculateDeltaW(int i, int j, double[] trainingData, double[] targets) {
        double[] errors = new double[ProcessMonitoring.lastOutputs.length];
        double errorSum = 0;
        for (int k = 0; k < ProcessMonitoring.lastOutputs.length; k++) {
            double lastOutputK = ProcessMonitoring.lastOutputs[k];
            errors[k] = (targets[k] - lastOutputK) * lastOutputK * (1 - lastOutputK) *
                    getWeightsOfOutputLayer()[k][i];
            errorSum += errors[k];
        }
        int numberOfWeights = hiddenLayer.get(i).getWeights().length - 1;
        double inputJ = (j < numberOfWeights) ? trainingData[j] : 1;
        double deltaW = errorSum * ProcessMonitoring.lastOutputsFromHiddenLayer[i] *
                (1 - ProcessMonitoring.lastOutputsFromHiddenLayer[i]) *
                inputJ * TrainingParameter.learningRate;
        return deltaW;
    }
}
