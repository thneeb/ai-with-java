package de.neebs.ai.control.rl.djl;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import de.neebs.ai.control.rl.ObservationImage;
import de.neebs.ai.control.rl.QNetwork;
import de.neebs.ai.control.rl.TrainingData;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class NeuralNetworkImage<O extends ObservationImage> extends AbstractDjlNetwork<O> {
    public NeuralNetworkImage(NeuralNetworkFactory factory, String filename, long seed) {
        super(factory, filename, seed);
    }

    public NeuralNetworkImage(NeuralNetworkFactory factory, long seed) {
        super(factory, seed);
    }

    @Override
    public double[] predict(O observation) {
        try (Predictor<BufferedImage, double[]> predictor = getModel().newPredictor(new Translator<>() {
            @Override
            public double[] processOutput(TranslatorContext ctx, NDList list) {
                float[] floats = list.singletonOrThrow().toFloatArray();
                return IntStream.range(0, floats.length).mapToDouble(f -> (double)floats[f]).toArray();
            }

            @Override
            public NDList processInput(TranslatorContext ctx, BufferedImage input) {
                return new NDList(transformImage(input));
            }
        })) {
            return predictor.predict(observation.getObservation());
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void train(O observation, double[] target) {
        throw new UnsupportedOperationException();
    }

    private NDArray transformImage(BufferedImage input) {
        Image djlImage = ImageFactory.getInstance().fromImage(input);
        NDArray array = djlImage.toNDArray(getManager(), Image.Flag.GRAYSCALE);
        array = array.toType(DataType.FLOAT32, false);
        array = array.transpose(2, 0, 1);
        return array;
    }

    private float[] convert(double [] target) {
        float[] floats = new float[target.length];
        for (int i = 0; i < target.length; i++) {
            floats[i] = (float)target[i];
        }
        return floats;
    }

    @Override
    public void train(List<TrainingData<O>> trainingData) {
        try (GradientCollector gc = getTrainer().newGradientCollector()) {
            List<NDArray> inputs = new ArrayList<>();
            List<NDArray> targets = new ArrayList<>();
            for (TrainingData<O> data : trainingData) {
                inputs.add(transformImage(data.getObservation().getObservation()));
                targets.add(getManager().create(convert(data.getOutput())));
            }

            Dataset ds = new ArrayDataset.Builder()
                    .setData(NDArrays.stack(new NDList(inputs)))
                    .optLabels(NDArrays.stack(new NDList(targets)))
                    .setSampling(32, true)
                    .build();
            for (Batch batch : ds.getData(getManager())) {
                NDList predictions = getTrainer().forward(batch.getData());
                NDArray loss = getTrainer().getLoss().evaluate(batch.getLabels(), predictions);
                gc.backward(loss);
                getTrainer().step();
            }
        } catch (IOException | TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public QNetwork<O> copy() {
        NeuralNetworkImage<O> copy = new NeuralNetworkImage<>(getFactory(), getSeed());
        copy.copyParams(this);
        return copy;
    }
}
