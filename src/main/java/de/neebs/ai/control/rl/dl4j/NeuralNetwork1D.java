package de.neebs.ai.control.rl.dl4j;

import de.neebs.ai.control.rl.Observation1D;
import de.neebs.ai.control.rl.QNetwork;
import de.neebs.ai.control.rl.TrainingData;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;
import java.util.Random;

public class NeuralNetwork1D<O extends Observation1D> extends AbstractDl4jNetwork<O> {
    public NeuralNetwork1D(NeuralNetworkFactory factory, long seed) {
        super(factory, seed);
    }

    public NeuralNetwork1D(String filename) {
        super(filename);
    }

    @Override
    public void copyParams(QNetwork<O> other) {
        super.copyParams((AbstractDl4jNetwork<O>) other);
    }

    @Override
    public void train(O observation, double[] target) {
        INDArray myInput = Nd4j.create(observation.getFlattenedObservation());
        myInput = myInput.reshape(1, observation.getFlattenedObservation().length);
        INDArray myOutput = Nd4j.create(target);
        myOutput = myOutput.reshape(1, target.length);
        getNetwork().fit(myInput, myOutput);
    }

    @Override
    public double[] predict(O observation) {
        INDArray myInput = Nd4j.create(observation.getFlattenedObservation());
        myInput = myInput.reshape(1, observation.getFlattenedObservation().length);
        INDArray myOutput = getNetwork().output(myInput);
        return myOutput.toDoubleVector();
    }

    public void train(TrainingData<O> trainingData) {
        train(List.of(trainingData));
    }

    @Override
    public void train(List<TrainingData<O>> trainingData) {
        double[][] inputs = trainingData.stream()
                .map(TrainingData::getInput)
                .map(Observation1D::getFlattenedObservation)
                .toArray(double[][]::new);
        double[][] outputs = trainingData.stream()
                .map(TrainingData::getOutput)
                .toArray(double[][]::new);
        DataSet dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        getNetwork().fit(dataSet);
    }

    public NeuralNetwork1D<O> copy() {
        return new NeuralNetwork1D<>((long seed) -> {
            MultiLayerNetwork n = getNetwork().clone();
            n.setParams(getNetwork().params());
            return n;
        }, new Random().nextLong());
    }
}
