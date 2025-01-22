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
import java.io.File;
import java.io.IOException;
import java.util.List;

public class NeuralNetworkImage {
    private final MultiLayerNetwork network;
    private final Java2DNativeImageLoader loader = new Java2DNativeImageLoader();

    public NeuralNetworkImage(NeuralNetworkFactory factory) {
        this.network = factory.createNeuralNetwork();
    }

    public NeuralNetworkImage(String filename) {
        try {
            this.network = MultiLayerNetwork.load(new File(filename), true);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public void copyParams(NeuralNetworkImage other) {
        network.setParams(other.network.params());
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

    public void save(String filename) {
        try {
            network.save(new File(filename), true);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Getter
    @Builder
    public static class TrainingData {
        private BufferedImage input;
        private double[] output;
    }
}
