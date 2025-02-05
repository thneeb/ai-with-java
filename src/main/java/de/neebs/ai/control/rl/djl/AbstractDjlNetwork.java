package de.neebs.ai.control.rl.djl;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.training.Trainer;
import de.neebs.ai.control.rl.Observation;
import de.neebs.ai.control.rl.QNetwork;
import lombok.Getter;

import java.io.IOException;
import java.nio.file.Path;

@Getter
public abstract class AbstractDjlNetwork<O extends Observation> implements QNetwork<O> {
    private final Model model;
    private final NDManager manager = NDManager.newBaseManager();
    private final Trainer trainer;

    public AbstractDjlNetwork(NeuralNetworkFactory factory, String filename, long seed) {
        try {
            model = factory.createNeuralNetwork(seed);
            model.load(Path.of(filename + "-djl"), model.getName());
            trainer = factory.createTrainer(model);
        } catch (IOException | MalformedModelException e) {
            throw new IllegalStateException(e);
        }
    }

    public AbstractDjlNetwork(NeuralNetworkFactory factory, long seed) {
        model = factory.createNeuralNetwork(seed);
        trainer = factory.createTrainer(model);
    }

    @Override
    public void save(String filename) {
        try {
            model.save(Path.of(filename + "-djl"), model.getName());
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void copyParams(QNetwork<O> other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public QNetwork<O> copy() {
        throw new UnsupportedOperationException();
    }
}
