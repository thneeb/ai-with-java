package de.neebs.ai.control.perceptron;

public class TrainingParameter {
    public static final int numberOfEpochs = 30;
    public static final double learningRate = 0.5;
    public static final ActivationFunction activationFunction = ActivationFunction.HEAVISIDE;
    public static final double[][] inputs = LogicalAndData.inputs;
    public static final double[] targets = LogicalAndData.targets;
}
