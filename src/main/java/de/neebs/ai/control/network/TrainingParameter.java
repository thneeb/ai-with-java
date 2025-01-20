package de.neebs.ai.control.network;

public class TrainingParameter {
    public static final int numberOfEpochs = 100000;
    public static final double learningRate = 0.1;
    public static final ActivationFunction activationFunction = ActivationFunction.SIGMOID;
    public static final double faultTolerance = 0.1;
    public static final double[][] inputs = LogicalAndData.inputs;
    public static final double[][] targets = LogicalAndData.targets;
    public static final double[][] weightsOfHiddenLayer = null;
    public static final double[][] weightsOfOutputLayer = null;
    public static final boolean isBiasBackPropagationDesired = true;
}
