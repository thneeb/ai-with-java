package de.neebs.ai.control.games;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import de.neebs.ai.control.rl.QNetwork;
import de.neebs.ai.control.rl.djl.NeuralNetworkFactory;
import de.neebs.ai.control.rl.djl.NeuralNetworkImage;

import java.util.Random;

public class PongDJL implements NeuralNetworkFactory {
    private final int height;
    private final int width;
    private final int channels;
    private final int actionSize;

    PongDJL(int width, int height, int channels, int actionSize) {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.actionSize = actionSize;
    }

    @Override
    public Model createNeuralNetwork(long seed) {
        Model model = Model.newInstance("pong-dqn", Device.cpu());

        SequentialBlock block = new SequentialBlock()
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(8, 8))
                        .optStride(new Shape(4, 4))
                        .setFilters(32)
                        .build())
                .add(f -> Activation.leakyRelu(f, 0.1f))
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(4, 4))
                        .optStride(new Shape(2, 2))
                        .setFilters(64)
                        .build())
                .add(f -> Activation.leakyRelu(f, 0.1f))
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optStride(new Shape(1, 1))
                        .setFilters(64)
                        .build())
                .add(f -> Activation.leakyRelu(f, 0.1f))
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(512).build())
                .add(f -> Activation.leakyRelu(f, 0.1f))
                .add(Linear.builder().setUnits(actionSize).build());
        block.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        block.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, channels, height, width));
        model.setBlock(block);
        return model;
    }

    @Override
    public Trainer createTrainer(Model model) {
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(Optimizer.adam().build());
        return model.newTrainer(config);
    }

    QNetwork<Pong.GameStateImage> createQNetwork(String filename) {
        if (filename == null) {
            return new NeuralNetworkImage<>(this, new Random().nextLong());
        } else {
            return new NeuralNetworkImage<>(this, filename, new Random().nextLong());
        }
    }
}
