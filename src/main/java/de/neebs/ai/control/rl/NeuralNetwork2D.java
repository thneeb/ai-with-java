package de.neebs.ai.control.rl;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Random;

public class NeuralNetwork2D<O extends Observation2D> extends AbstractDl4jNetwork<O> {
    public NeuralNetwork2D(NeuralNetworkFactory factory, long seed) {
        super(factory, seed);
    }

    public NeuralNetwork2D(String filename) {
        super(filename);
    }

    @Override
    public void train(O observation, double[] target) {
        INDArray myInput = Nd4j.create(new double[][][] { observation.getObservation() });
        myInput = myInput.reshape(1, 1, observation.getObservation().length, observation.getObservation()[0].length);
        INDArray myOutput = Nd4j.create(target);
        myOutput = myOutput.reshape(1, target.length);
        getNetwork().fit(myInput, myOutput);
    }

    @Override
    public double[] predict(O observation) {
        INDArray myInput = Nd4j.create(new double[][][] { observation.getObservation() });
        myInput = myInput.reshape(1, 1, observation.getObservation().length, observation.getObservation()[0].length);
        INDArray myOutput = getNetwork().output(myInput);
        return myOutput.toDoubleVector();
    }

    public void train(List<TrainingData<O>> trainingData) {
        double[][][][] inputs = trainingData.stream()
                .map(TrainingData::getInput)
                .map(Observation2D::getObservation)
                .map(f -> new double[][][] { f })
                .toArray(double[][][][]::new);
        double[][] outputs = trainingData.stream()
                .map(TrainingData::getOutput)
                .toArray(double[][]::new);
        DataSet dataSet = new DataSet(
                Nd4j.create(inputs),
                Nd4j.create(outputs));
        getNetwork().fit(dataSet);
    }

    public NeuralNetwork2D<O> copy() {
        return new NeuralNetwork2D<>((long seed) -> {
            MultiLayerNetwork n = getNetwork().clone();
            n.setParams(getNetwork().params());
            return n;
        }, new Random().nextLong());
    }

    @Override
    public void copyParams(QNetwork<O> other) {
        super.copyParams((AbstractDl4jNetwork<O>) other);
    }
}
