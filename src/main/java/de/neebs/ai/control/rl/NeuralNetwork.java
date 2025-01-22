package de.neebs.ai.control.rl;

public interface NeuralNetwork<O extends Observation> {
    void train(O observation, double[] target);

    double[] predict(O observation);

    void save(String filename);
}
