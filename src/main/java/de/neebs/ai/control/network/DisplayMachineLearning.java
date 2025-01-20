package de.neebs.ai.control.network;

public class DisplayMachineLearning {
    public static void show(double[][] input, double[][] target) {
        System.out.println("\ninputs and targets:");
        for (int i = 0; i < input.length; i++) {
            System.out.print("input " + (i + 1) + ": ");
            for (int j = 0; j < input[0].length; j++)
                System.out.print(input[i][j] + " ");
            System.out.print("    target: ");
            for (int j = 0; j < target[0].length; j++)
                System.out.print(target[i][j] + " ");
            System.out.println();
        }
    }

    public static void showInput(double[] input) {
        showDataWithComment("\ninput: ", input);
    }

    public static void showOutput(double[] output) {
        showDataWithComment("\noutput: ", output);
    }

    public static void showTarget(double[] target) {
        showDataWithComment("target: ", target);
    }

    private static void showDataWithComment(String comment, double[] data) {
        System.out.println(comment);
        for (int i = 0; i < data.length; i++)
            System.out.print(data[i] + " ");
        System.out.println();
    }

    public static void showWeights(double[][] weightsOfInputLayer,
                                   double[][] weightsOfOutputLayer) {
        showWeightsWithComment("weights and bias of hidden layer:", weightsOfInputLayer);
        showWeightsWithComment("weights and bias of output layer:", weightsOfOutputLayer);
    }

    private static void showWeightsWithComment(String comment, double[][] weights) {
        System.out.println("\n" + comment);
        for (int i = 0; i < weights.length; i++) {
            System.out.print("neuron " + (i + 1) + ": ");
            for (int j = 0; j < weights[i].length; j++)
                System.out.print(weights[i][j] + " ");
            System.out.println();
        }
    }
}
