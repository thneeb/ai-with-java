package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class NeuralNetwork {
    private final MultiLayerNetwork network;

    public NeuralNetwork(NeuralNetworkFactory factory) {
        this.network = factory.createNeuralNetwork();
    }

    public void copyParams(NeuralNetwork other) {
        network.setParams(other.network.params());
    }

    public void train(double[] input, double[] output) {
        INDArray myInput = Nd4j.create(input);
        myInput = myInput.reshape(1, input.length);
        INDArray myOutput = Nd4j.create(output);
        myOutput = myOutput.reshape(1, output.length);
        network.fit(myInput, myOutput);
    }

    public void train(TrainingData trainingData) {
        train(List.of(trainingData));
    }

    public void train(List<TrainingData> trainingData) {
        double[][] inputs = trainingData.stream()
                .map(TrainingData::getInput)
                .toArray(double[][]::new);
        double[][] outputs = trainingData.stream()
                .map(TrainingData::getOutput)
                .toArray(double[][]::new);
        DataSet dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        network.fit(dataSet);
    }

    public double[] predict(double[] input) {
        INDArray myInput = Nd4j.create(input);
        myInput = myInput.reshape(1, input.length);
        INDArray myOutput = network.output(myInput);
        return myOutput.toDoubleVector();
    }

    public NeuralNetwork copy() {
        return new NeuralNetwork(() -> {
            MultiLayerNetwork n = network.clone();
            n.setParams(network.params());
            return n;
        });
    }

    public void setListeners(TrainingListener... listeners) {
        network.setListeners(listeners);
    }

    @Getter
    @Builder
    public static class TrainingData {
        private double[] input;
        private double[] output;
    }
}
