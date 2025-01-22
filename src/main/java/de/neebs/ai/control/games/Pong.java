package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import de.neebs.ai.control.rl.gym.GymClient;
import de.neebs.ai.control.rl.gym.GymEnvironment;
import de.neebs.ai.control.rl.gym.ObservationWrapper;
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
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

@Slf4j
@Service
@RequiredArgsConstructor
public class Pong {
    private final GymClient gymClient;

    enum GameAction {
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

        private double getChannelScale() {
            return Stream.of(observation).flatMap(Stream::of).flatMapToDouble(DoubleStream::of).max().orElse(0);
        }

        @Override
        public double[] getFlattenedObservation() {
            throw new UnsupportedOperationException();
        }
    }

    @Getter
    @RequiredArgsConstructor
    static class GameStateImage implements ObservationImage {
        private final BufferedImage observation;

        @Override
        public double[] getFlattenedObservation() {
            return new double[0];
        }
    }

    private static class Utils {
        private Utils() {
        }

        private static void saveImage(GameState3D state, String name) {
            BufferedImage image = createImage(state);
            try {
                File outputFile = new File(name);
                ImageIO.write(image, "bmp", outputFile);
            } catch (IOException e) {
                throw new IllegalStateException();
            }
        }

        private static BufferedImage convertToGrayscale(BufferedImage rgbImage) {
            int width = rgbImage.getWidth();
            int height = rgbImage.getHeight();

            // Erstelle ein neues Graustufenbild
            BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

            Graphics2D g = grayImage.createGraphics();
            g.drawImage(rgbImage, 0, 0, null);
            g.dispose();

            return grayImage;
        }

        private static BufferedImage resizeImage(BufferedImage image, int newWidth, int newHeight) {
            Image tmp = image.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
            BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_BYTE_GRAY);

            Graphics2D g2d = resizedImage.createGraphics();
            g2d.drawImage(tmp, 0, 0, null);
            g2d.dispose();

            return resizedImage;
        }

        private static GameState3D scaleObservation(GameState3D state) {
            double maxChannelScale = 255;
            for (int col = 0; col < state.getWidth(); col++) {
                for (int row = 0; row < state.getHeight(); row++) {
                    for (int channel = 0; channel < state.getChannels(); channel++) {
                        state.observation[col][row][channel] /= maxChannelScale;
                    }
                }
            }
            return state;
        }

        private static BufferedImage createImage(GameState3D state) {
            double maxChannelScale = state.getChannelScale();
            BufferedImage image = new BufferedImage(state.getWidth(), state.getHeight(), BufferedImage.TYPE_INT_RGB );
            for (int col = 0; col < state.getWidth(); col++) {
                for (int row = 0; row < state.getHeight(); row++) {
                    int rgb;
                    if (state.getChannels() == 3) {
                        rgb = ((int)state.observation[col][row][0] << 16) | ((int)state.observation[col][row][1] << 8) | (int)state.observation[col][row][2];
                    } else {
                        if (maxChannelScale > 1) {
                            rgb = (int)(state.observation[col][row][0]);
                        } else {
                            rgb = (int)(state.observation[col][row][0]) * 255;
                        }
                        rgb = (rgb << 16) | (rgb << 8) | rgb;
                    }
                    image.setRGB(col, row, rgb);
                }
            }
            return image;
        }

        private static GameState3D createObservation(BufferedImage image) {
            GameState3D observation = new GameState3D();
            observation.setObservation(new double[image.getWidth()][image.getHeight()][1]);
            for (int col = 0; col < image.getWidth(); col++) {
                for (int row = 0; row < image.getHeight(); row++) {
                    int rgb = image.getRGB(col, row);
                    observation.observation[col][row][0] = rgb & 0xFF;
                }
            }
            return observation;
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
        public MultiLayerNetwork createNeuralNetwork() {
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
        Environment<GameAction, GameState3D> env3d =
                new GymEnvironment<>(GameAction.class, GameState3D.class, gymClient).init("ale_py:ALE/Pong-v5");
        Environment<GameAction, GameStateImage> envImage = new ReduceImageSize(env3d);

        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(0.42).epsilonMin(0.1).decreaseRate(0.01).build();

        NeuralNetworkImage network;
        if (startFresh) {
            network = new NeuralNetworkImage(new MyNeuralNetworkFactory());
        } else {
            network = new NeuralNetworkImage("pong-agent.net");
        }

        Agent<GameAction, GameStateImage> agent = new QLearningAgentImage<>(
                network,
                greedy,
                0.99);

        SinglePlayerGame<GameAction, GameStateImage, Environment<GameAction, GameStateImage>> game = new SinglePlayerGame<>(envImage, agent);
        for (int i = 0; i < 500; i++) {
            PlayResult<GameStateImage> result = game.play();
            greedy.decrementEpsilon(i);
            network.save("pong-agent.net");
            log.info("Runde: {}, Reward: {}, Epsilon: {}, Frames: {}", i, result.getReward(), greedy.getEpsilon(), result.getRounds());
        }
/*
        GameState state = env.reset();
        boolean done = false;
        int i = 0;
        while (!done) {
            String imageName = "output-" + String.format("%05d", i) + ".bmp";
            Utils.saveImage(state, imageName);
            StepResult<GameState> stepResult = env.step(GameAction.NOOP);
            state = stepResult.getObservation();
            done = stepResult.isDone();
            i++;
        }
 */
    }
}
