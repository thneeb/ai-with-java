package de.neebs.ai.control.rl.dl4j;

import de.neebs.ai.control.rl.Action;
import de.neebs.ai.control.rl.ObservationImage;
import de.neebs.ai.control.rl.QNetwork;
import de.neebs.ai.control.rl.TrainingData;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class NeuralNetworkImage<A extends Action, O extends ObservationImage> extends AbstractDl4jNetwork<A, O> {
    private final Java2DNativeImageLoader loader = new Java2DNativeImageLoader();

    public NeuralNetworkImage(NeuralNetworkFactory factory, long seed) {
        super(factory, seed);
    }

    public NeuralNetworkImage(String filename) {
        super(filename);
    }

    @Override
    public void train(O observation, double[] target) {
        INDArray input = asMatrix(observation.getObservation());
        INDArray output = Nd4j.create(target);
        getNetwork().fit(new DataSet(input, output));
    }

    @Override
    public double[] predict(O observation) {
        try {
            INDArray myInput = loader.asMatrix(observation.getObservation());
            new ImagePreProcessingScaler().transform(myInput);
            INDArray myOutput = getNetwork().output(myInput);
            return myOutput.toDoubleVector();
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public void train(List<TrainingData<O>> trainingData) {
        INDArray[] inputs = trainingData.stream()
                .map(TrainingData::getObservation)
                .map(ObservationImage::getObservation)
                .map(this::asMatrix)
                .toArray(INDArray[]::new);
        double[][] outputs = trainingData.stream()
                .map(TrainingData::getOutput)
                .toArray(double[][]::new);
        DataSet dataSet = new DataSet(Nd4j.concat(0, inputs), Nd4j.create(outputs), null, null);
        getNetwork().fit(dataSet);
    }

    private INDArray asMatrix(BufferedImage input) {
        try (INDArray array = loader.asMatrix(input)){
            return array.div(255);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public QNetwork<A, O> copy() {
        return new NeuralNetworkImage<>((long seed) -> {
            MultiLayerNetwork n = getNetwork().clone();
            n.setParams(getNetwork().params());
            return n;
        }, new Random().nextLong());
    }
}
