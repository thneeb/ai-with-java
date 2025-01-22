package de.neebs.ai.control.rl;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class NeuralNetwork1D<O extends Observation1D> implements NeuralNetwork<O> {
    private final MultiLayerNetwork network;

    public NeuralNetwork1D(NeuralNetworkFactory factory) {
        this.network = factory.createNeuralNetwork();
    }

    public void copyParams(NeuralNetwork1D<O> other) {
        network.setParams(other.network.params());
    }

    @Override
    public void train(O observation, double[] target) {
        INDArray myInput = Nd4j.create(observation.getFlattenedObservation());
        myInput = myInput.reshape(1, observation.getFlattenedObservation().length);
        INDArray myOutput = Nd4j.create(target);
        myOutput = myOutput.reshape(1, target.length);
        network.fit(myInput, myOutput);
    }

    @Override
    public double[] predict(O observation) {
        INDArray myInput = Nd4j.create(observation.getFlattenedObservation());
        myInput = myInput.reshape(1, observation.getFlattenedObservation().length);
        INDArray myOutput = network.output(myInput);
        return myOutput.toDoubleVector();
    }

    @Override
    public void save(String filename) {

    }

    public void train(double[] input, double[] output) {
        INDArray myInput = Nd4j.create(input);
        myInput = myInput.reshape(1, input.length);
        INDArray myOutput = Nd4j.create(output);
        myOutput = myOutput.reshape(1, output.length);
        network.fit(myInput, myOutput);
    }

    public void train(TrainingData<O> trainingData) {
        train(List.of(trainingData));
    }

    public void train(List<TrainingData<O>> trainingData) {
        double[][] inputs = trainingData.stream()
                .map(TrainingData::getInput)
                .map(Observation1D::getFlattenedObservation)
                .toArray(double[][]::new);
        double[][] outputs = trainingData.stream()
                .map(TrainingData::getOutput)
                .toArray(double[][]::new);
        DataSet dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        network.fit(dataSet);
    }

    public NeuralNetwork1D<O> copy() {
        return new NeuralNetwork1D<>(() -> {
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
    @AllArgsConstructor
    public static class TrainingData<O extends Observation> {
        private O input;
        private double[] output;
    }
}
