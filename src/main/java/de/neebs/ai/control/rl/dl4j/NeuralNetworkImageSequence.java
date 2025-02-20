package de.neebs.ai.control.rl.dl4j;

import de.neebs.ai.control.rl.*;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetworkImageSequence<A extends Action, O extends ObservationImageSequence> extends AbstractDl4jNetwork<A, O> {
    private final Java2DNativeImageLoader loader = new Java2DNativeImageLoader();

    public NeuralNetworkImageSequence(NeuralNetworkFactory factory, long seed) {
        super(factory, seed);
    }

    public NeuralNetworkImageSequence(String filename) {
        super(filename);
    }

    @Override
    public void train(O observation, double[] target) {
        List<INDArray> list = new ArrayList<>();
        for (BufferedImage image : observation.getObservation()) {
            list.add(asMatrix(image));
        }
        INDArray input = Nd4j.concat(0, list.toArray(new INDArray[0]));
        INDArray output = Nd4j.create(target);
        getNetwork().fit(new DataSet(input, output));
    }

    @Override
    public double[] predict(O observation) {
        INDArray input = asMatrix(observation.getObservation());
        INDArray myOutput = getNetwork().output(input);
        return myOutput.toDoubleVector();
    }

    public void train(List<TrainingData<O>> trainingData) {
        INDArray[] inputs = trainingData.stream()
                .map(TrainingData::getObservation)
                .map(ObservationImageSequence::getObservation)
                .map(this::asMatrix)
                .toArray(INDArray[]::new);
        double[][] outputs = trainingData.stream()
                .map(TrainingData::getOutput)
                .toArray(double[][]::new);
        DataSet dataSet = new DataSet(Nd4j.concat(0, inputs), Nd4j.create(outputs), null, null);
        getNetwork().fit(dataSet);
    }

    @Override
    public boolean isFastTrainingSupported() {
        return true;
    }

    @Override
    public void train(List<Transition<A, O>> transitions, double gamma) {
        try (INDArray rewards = Nd4j.create(transitions.stream().mapToDouble(Transition::getReward).toArray())) {
            INDArray[] states = transitions.stream()
                    .map(Transition::getObservation)
                    .map(ObservationImageSequence::getObservation)
                    .map(this::asMatrix).toArray(INDArray[]::new);
            INDArray[] nextStates = transitions.stream()
                    .map(Transition::getNextObservation)
                    .map(ObservationImageSequence::getObservation)
                    .map(this::asMatrix).toArray(INDArray[]::new);
            INDArray qValues = getNetwork().output(Nd4j.concat(0, states));
            INDArray nextQValues = getNetwork().output(Nd4j.concat(0, nextStates));
            INDArray maxNextQ = nextQValues.max(1);
            INDArray targetQValues = rewards.add(maxNextQ.mul(gamma));
            for (int i = 0; i < transitions.size(); i++) {
                qValues.putScalar(i, transitions.get(i).getAction().ordinal(), targetQValues.getDouble(i));
            }
            getNetwork().fit(new DataSet(Nd4j.concat(0, states), qValues));
        }
    }

    private INDArray asMatrix(List<BufferedImage> images) {
        List<INDArray> list = new ArrayList<>();
        for (BufferedImage image : images) {
            list.add(asMatrix(image));
        }
        return Nd4j.concat(1, list.toArray(new INDArray[0]));
    }

    private INDArray asMatrix(BufferedImage image) {
        try (INDArray array = loader.asMatrix(image)) {
            return array.div(255);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public QNetwork<A, O> copy() {
        return new NeuralNetworkImageSequence<>((long seed) -> {
            MultiLayerNetwork n = getNetwork().clone();
            n.setParams(getNetwork().params());
            return n;
        }, new Random().nextLong());
    }
}
