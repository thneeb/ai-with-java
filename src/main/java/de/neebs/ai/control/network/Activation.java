package de.neebs.ai.control.network;

public class Activation {
    public static double activateWithHeaviside(double sum) {
        if (sum > 0)
            return 1;
        return 0;
    }

    public static double activateWithSigmoid(double sum) {
        return (1 / (1 + Math.exp(-sum)));
    }
}
