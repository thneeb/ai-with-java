package de.neebs.ai.control.rl.djl;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.training.Trainer;
import de.neebs.ai.control.rl.Observation;
import de.neebs.ai.control.rl.QNetwork;
import lombok.Getter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

@Getter
public abstract class AbstractDjlNetwork<O extends Observation> implements QNetwork<O> {
    private final NeuralNetworkFactory factory;
    private final Model model;
    private final NDManager manager = NDManager.newBaseManager();
    private final Trainer trainer;
    private final long seed;

    public AbstractDjlNetwork(NeuralNetworkFactory factory, String filename, long seed) {
        try {
            this.seed = seed;
            this.factory = factory;
            model = factory.createNeuralNetwork(seed);
            model.load(Path.of(filename + "-djl"), model.getName());
            trainer = factory.createTrainer(model);
        } catch (IOException | MalformedModelException e) {
            throw new IllegalStateException(e);
        }
    }

    public AbstractDjlNetwork(NeuralNetworkFactory factory, long seed) {
        this.seed = seed;
        this.factory = factory;
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
    public void copyParams(QNetwork<O> source) {
        // Angenommen, beide Modelle haben denselben Block (z.B. denselben Architekturbaum)
        Block sourceBlock = ((AbstractDjlNetwork<O>) source).model.getBlock();
        Block targetBlock = model.getBlock();

        // Hole die Parameter als Map (Name -> Parameter) aus dem Quellblock
        ParameterList sourceParams = sourceBlock.getParameters();

        // Iteriere über alle Parameter des Zielblocks und setze deren Werte
        for (Map.Entry<String, Parameter> entry : targetBlock.getParameters().toMap().entrySet()) {
            String paramName = entry.getKey();
            Parameter targetParam = entry.getValue();

            // Wenn ein entsprechender Parameter im Quellmodell existiert, übernehme ihn
            if (sourceParams.contains(paramName)) {
                Parameter sourceParam = sourceParams.get(paramName);
                targetParam.getArray().set(sourceParam.getArray().toFloatArray());
            }
        }
    }
}
