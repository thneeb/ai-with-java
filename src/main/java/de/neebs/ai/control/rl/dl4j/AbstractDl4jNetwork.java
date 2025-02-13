package de.neebs.ai.control.rl.dl4j;

import de.neebs.ai.control.rl.Observation;
import de.neebs.ai.control.rl.QNetwork;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import java.io.File;
import java.io.IOException;

@Getter(value = AccessLevel.PROTECTED)
@RequiredArgsConstructor
public abstract class AbstractDl4jNetwork<O extends Observation> implements QNetwork<O> {
    private final MultiLayerNetwork network;

    public AbstractDl4jNetwork(NeuralNetworkFactory factory, long seed) {
        this.network = factory.createNeuralNetwork(seed);
    }

    public AbstractDl4jNetwork(String filename) {
        try {
            this.network = MultiLayerNetwork.load(new File(filename + "-dl4j.zip"), true);
            network.setListeners(new ScoreIterationListener(100));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void copyParams(QNetwork<O> source) {
        network.setParams(((AbstractDl4jNetwork<O>)source).network.params());
    }

    @Override
    public void save(String filename) {
        try {
            network.save(new File(filename + "-dl4j.zip"));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
