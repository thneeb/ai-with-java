package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

public class NeuralNetworkImage {
    private final MultiLayerNetwork network;
    private final Java2DNativeImageLoader loader = new Java2DNativeImageLoader();

    public NeuralNetworkImage(NeuralNetworkFactory factory) {
        this.network = factory.createNeuralNetwork();
    }

    public void copyParams(NeuralNetworkImage other) {
        network.setParams(other.network.params());
    }

    public void train(double[][][] input, double[] output) {
        INDArray myInput = Nd4j.create(input);
        INDArray myOutput = Nd4j.create(output);
        network.fit(myInput, myOutput);
    }

    public void train(TrainingData trainingData) {
        train(List.of(trainingData));
    }

    public void train(List<TrainingData> trainingData) {
        INDArray[] inputs = trainingData.stream()
                .map(TrainingData::getInput)
                .map(this::asMatrix)
                .toArray(INDArray[]::new);
        double[][] outputs = trainingData.stream()
                .map(TrainingData::getOutput)
                .toArray(double[][]::new);
        DataSet dataSet = new DataSet(Nd4j.concat(0, inputs), Nd4j.create(outputs));
        network.fit(dataSet);
    }

    private INDArray asMatrix(BufferedImage input) {
        try {
            return loader.asMatrix(input);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public double[] predict(BufferedImage input) {
        try {
            INDArray myInput = loader.asMatrix(input);
            INDArray myOutput = network.output(myInput);
            return myOutput.toDoubleVector();
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public NeuralNetworkImage copy() {
        return new NeuralNetworkImage(() -> {
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
        private BufferedImage input;
        private double[] output;
    }
}
