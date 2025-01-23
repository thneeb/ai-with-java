package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class NeuralNetworkImage<O extends ObservationImage> implements NeuralNetwork<O> {
    private final MultiLayerNetwork network;
    private final Java2DNativeImageLoader loader = new Java2DNativeImageLoader();

    public NeuralNetworkImage(NeuralNetworkFactory factory) {
        this.network = factory.createNeuralNetwork(new Random().nextLong());
    }

    public NeuralNetworkImage(String filename) {
        try {
            this.network = MultiLayerNetwork.load(new File(filename), true);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void train(O observation, double[] target) {
        INDArray input = asMatrix(observation.getObservation());
        INDArray output = Nd4j.create(target);
        network.fit(new DataSet(input, output));
    }

    @Override
    public double[] predict(O observation) {
        try {
            INDArray myInput = loader.asMatrix(observation.getObservation());
            INDArray myOutput = network.output(myInput);
            return myOutput.toDoubleVector();
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public void train(List<TrainingData<O>> trainingData) {
        INDArray[] inputs = trainingData.stream()
                .map(TrainingData::getInput)
                .map(ObservationImage::getObservation)
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

    public void save(String filename) {
        try {
            network.save(new File(filename), true);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Getter
    @Builder
    public static class TrainingData<O extends ObservationImage> {
        private O input;
        private double[] output;
    }
}
