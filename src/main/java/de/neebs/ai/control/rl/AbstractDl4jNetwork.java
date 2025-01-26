package de.neebs.ai.control.rl;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import java.io.File;
import java.io.IOException;
import java.util.Random;

@Getter(value = AccessLevel.PROTECTED)
@RequiredArgsConstructor
public abstract class AbstractDl4jNetwork<O extends Observation> implements QNetwork<O> {
    private final MultiLayerNetwork network;

    public AbstractDl4jNetwork(NeuralNetworkFactory factory, long seed) {
        this.network = factory.createNeuralNetwork(seed);
    }

    public AbstractDl4jNetwork(String filename) {
        try {
            this.network = MultiLayerNetwork.load(new File(filename), true);
            network.setListeners(new ScoreIterationListener(1000));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public void copyParams(AbstractDl4jNetwork<O> other) {
        network.setParams(other.network.params());
    }

    @Override
    public void save(String filename) {
        try {
            network.save(new File(filename));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
