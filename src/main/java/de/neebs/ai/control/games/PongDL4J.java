package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.QNetwork;
import de.neebs.ai.control.rl.dl4j.NeuralNetworkFactory;
import de.neebs.ai.control.rl.dl4j.NeuralNetworkImage;
import de.neebs.ai.control.rl.dl4j.NeuralNetworkImageSequence;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

@Slf4j
public class PongDL4J implements NeuralNetworkFactory {
    private final int width;
    private final int height;
    private final int channels;
    private final int actionSize;

    public PongDL4J(int width, int height, int channels, int actionSize) {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.actionSize = actionSize;
    }

    @Override
    public MultiLayerNetwork createNeuralNetwork(long seed) {
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(4.0)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.001))
                .miniBatch(true)
                .list()
                .layer(new ConvolutionLayer.Builder(8, 8)
                        .stride(4, 4)
                        .nIn(channels)
                        .nOut(32)
                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.RELU_UNIFORM)
                        .build())
//                    .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder(4, 4)
                        .stride(2, 2)
                        .nOut(64)
                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.RELU_UNIFORM)
                        .build())
//                    .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.RELU_UNIFORM)
                        .build())
//                    .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .nOut(512)
                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(actionSize) // Ausgabe (z.B. 10 Klassen)
                        .activation(Activation.IDENTITY) // No transformation
//                        .weightInit(WeightInit.NORMAL)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels)) // Input-Shape definieren
                .build());
        model.init();
        log.info(model.summary());
        model.setListeners(new ScoreIterationListener(100));
        return model;
    }

    QNetwork<Pong.GameAction, Pong.GameStateImage> createQNetwork(String filename) {
        if (filename == null) {
            return new NeuralNetworkImage<>(this, new Random().nextLong());
        } else {
            return new NeuralNetworkImage<>(filename);
        }
    }

    QNetwork<Pong.GameAction, Pong.GameStateImageSequence> createQNetwork2(String filename) {
        if (filename == null) {
            return new NeuralNetworkImageSequence<>(this, new Random().nextLong());
        } else {
            return new NeuralNetworkImageSequence<>(filename);
        }
    }

}
