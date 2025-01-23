package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import de.neebs.ai.control.rl.gym.GymClient;
import de.neebs.ai.control.rl.gym.GymEnvironment;
import de.neebs.ai.control.rl.ObservationWrapper;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class Pong {
    private final GymClient gymClient;

    enum GameAction implements Action {
        NOOP,
        FIRE,
        RIGHT,
        LEFT,
        RIGHT_FIRE,
        LEFT_FIRE
    }

    @Getter
    @Setter
    public static class GameState3D implements Observation3D {
        private double[][][] observation;

        private int getWidth() {
            return observation.length;
        }

        private int getHeight() {
            return observation[0].length;
        }

        private int getChannels() {
            if (((Object)observation[0][0]).getClass().isArray()) {
                return observation[0][0].length;
            } else {
                return 1;
            }
        }
    }

    @Getter
    @RequiredArgsConstructor
    static class GameStateImage implements ObservationImage {
        private final BufferedImage observation;
    }

    static class Utils {
        private Utils() {
        }

        static void saveImage(GameState3D state, String name) {
            BufferedImage image = createImage(state);
            try {
                File outputFile = new File(name);
                ImageIO.write(image, "bmp", outputFile);
            } catch (IOException e) {
                throw new IllegalStateException();
            }
        }

        static BufferedImage convertToGrayscale(BufferedImage rgbImage) {
            int width = rgbImage.getWidth();
            int height = rgbImage.getHeight();

            // Erstelle ein neues Graustufenbild
            BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

            Graphics2D g = grayImage.createGraphics();
            g.drawImage(rgbImage, 0, 0, null);
            g.dispose();

            return grayImage;
        }

        static BufferedImage resizeImage(BufferedImage image, int newWidth, int newHeight) {
            Image tmp = image.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
            BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_BYTE_GRAY);

            Graphics2D g2d = resizedImage.createGraphics();
            g2d.drawImage(tmp, 0, 0, null);
            g2d.dispose();

            return resizedImage;
        }

        private static BufferedImage createImage(GameState3D state) {
            BufferedImage image = new BufferedImage(state.getWidth(), state.getHeight(), BufferedImage.TYPE_INT_RGB );
            for (int col = 0; col < state.getWidth(); col++) {
                for (int row = 0; row < state.getHeight(); row++) {
                    int rgb;
                    if (state.getChannels() == 3) {
                        rgb = ((int)state.observation[col][row][0] << 16) | ((int)state.observation[col][row][1] << 8) | (int)state.observation[col][row][2];
                    } else {
                        rgb = (int)(state.observation[col][row][0]);
                        rgb = (rgb << 16) | (rgb << 8) | rgb;
                    }
                    image.setRGB(col, row, rgb);
                }
            }
            return image;
        }
    }

    static class ReduceImageSize extends ObservationWrapper<GameAction, GameState3D, GameStateImage> {
        private static final int WIDTH = 84;
        private static final int HEIGHT = 84;

        ReduceImageSize(Environment<GameAction, GameState3D> env) {
            super(env);
        }

        @Override
        protected GameStateImage wrapper(GameState3D observation) {
            BufferedImage image = Utils.createImage(observation);
            image = Utils.convertToGrayscale(image);
            image = Utils.resizeImage(image, WIDTH, HEIGHT);
            return new GameStateImage(image);
        }

        @Override
        public List<Integer> getObservationSpace() {
            return List.of(WIDTH, HEIGHT, 1);
        }
    }

    static class MyNeuralNetworkFactory implements NeuralNetworkFactory {
        @Override
        public MultiLayerNetwork createNeuralNetwork(long seed) {
            MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Adam(0.0001))
                    .list()
                    .layer(new ConvolutionLayer.Builder(8, 8)
                            .stride(4, 4)
                            .nIn(1)
                            .nOut(32)
                            .weightInit(WeightInit.XAVIER)
                            .activation(Activation.RELU)
                            .build())
                    .layer(new ConvolutionLayer.Builder(4, 4)
                            .stride(2, 2)
                            .nIn(1)
                            .nOut(64)
                            .weightInit(WeightInit.XAVIER)
                            .activation(Activation.RELU)
                            .build())
                    .layer(new ConvolutionLayer.Builder(3, 3)
                            .stride(1, 1)
                            .nIn(1)
                            .nOut(64)
                            .weightInit(WeightInit.XAVIER)
                            .activation(Activation.RELU)
                            .build())
                    .layer(3, new DenseLayer.Builder()
                            .nOut(512)
                            .weightInit(WeightInit.XAVIER)
                            .activation(Activation.RELU)
                            .build())
                    .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(GameAction.values().length) // Ausgabe (z.B. 10 Klassen)
                            .activation(Activation.SOFTMAX) // Softmax für Klassifikation
                            .build())
                    .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutionalFlat(84, 84, 1)) // Input-Shape definieren
                    .build());
            model.init();
            log.info(model.summary());
            return model;
        }
    }

    public void execute(boolean startFresh) {
        String filename = "pong-agent.zip";
        Environment<GameAction, GameState3D> env3d =
                new GymEnvironment<>(GameAction.class, GameState3D.class, gymClient).init("ale_py:ALE/Pong-v5");
        Environment<GameAction, GameStateImage> envImage = new ReduceImageSize(env3d);

        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(0.42).epsilonMin(0.1).decreaseRate(0.01).build();

        NeuralNetworkImage<GameStateImage> network;
        if (startFresh) {
            network = new NeuralNetworkImage<>(new MyNeuralNetworkFactory());
        } else {
            network = new NeuralNetworkImage<>(filename);
        }

        Agent<GameAction, GameStateImage> agent = new QLearningAgentImage<>(
                network,
                greedy,
                0.99);

        SinglePlayerGame<GameAction, GameStateImage, Environment<GameAction, GameStateImage>> game = new SinglePlayerGame<>(envImage, agent);
        for (int i = 0; i < 500; i++) {
            PlayResult<GameStateImage> result = game.play();
            greedy.decrease(i);
            network.save(filename);
            log.info("Runde: {}, Reward: {}, Epsilon: {}, Frames: {}", i, result.getReward(), greedy.getEpsilon(), result.getRounds());
        }
    }
}
