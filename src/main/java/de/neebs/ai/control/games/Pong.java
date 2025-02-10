package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import de.neebs.ai.control.rl.gym.GymClient;
import de.neebs.ai.control.rl.gym.GymEnvironment;
import de.neebs.ai.control.rl.ObservationWrapper;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import javax.imageio.stream.FileImageOutputStream;
import javax.imageio.stream.ImageOutputStream;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class Pong {
    private static final int WIDTH = 84;
    private static final int HEIGHT = 84;
    private static final int CHANNELS = 1;

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
            BufferedImage image = new BufferedImage(state.getHeight(), state.getWidth(), BufferedImage.TYPE_INT_RGB );
            for (int col = 0; col < state.getWidth(); col++) {
                for (int row = 0; row < state.getHeight(); row++) {
                    int rgb;
                    if (state.getChannels() == 3) {
                        rgb = ((int)state.observation[col][row][0] << 16) | ((int)state.observation[col][row][1] << 8) | (int)state.observation[col][row][2];
                    } else {
                        rgb = (int)(state.observation[col][row][0]);
                        rgb = (rgb << 16) | (rgb << 8) | rgb;
                    }
                    image.setRGB(row, col, rgb);
                }
            }
            return image;
        }
    }

    static class ReduceImageSize extends ObservationWrapper<GameAction, GameState3D, GameStateImage> {
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

    static class ScaleReward extends RewardFitter<GameAction, GameStateImage> {
        ScaleReward(Environment<GameAction, GameStateImage> environment) {
            super(environment);
        }

        @Override
        protected double fitReward(double reward) {
            return reward;
        }
    }

    public void execute(boolean startFresh, boolean saveModel, Double epsilon, Integer startingEpisode, Integer episodes) {
        String filename = "pong-agent";
        Environment<GameAction, GameState3D> env3d =
                new GymEnvironment<>(GameAction.class, GameState3D.class, gymClient).init("ale_py:ALE/Pong-v5");
        Environment<GameAction, GameStateImage> envImage = new ReduceImageSize(env3d);
        envImage = new ScaleReward(envImage);

        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder()
                .epsilon(epsilon == null ? 1.0 : epsilon)
                .epsilonMin(0.1)
                .decreaseRate(0.01)
                .build();

//        QNetwork<GameStateImage> network = new PongDL4J(WIDTH, HEIGHT, CHANNELS, envImage.getActionSpace().getActions().size()).createQNetwork(startFresh ? null : filename);
        QNetwork<GameStateImage> network = new PongDJL(WIDTH, HEIGHT, CHANNELS, envImage.getActionSpace().getActions().size()).createQNetwork(startFresh ? null : filename);

        Agent<GameAction, GameStateImage> agent = new DoubleQLearningAgent<>(
                network,
                greedy,
                0.99,
                1000);

        SinglePlayerGame<GameAction, GameStateImage, Environment<GameAction, GameStateImage>> game = new SinglePlayerGame<>(envImage, agent, 10000, 64, 0.02);
        for (int i = (startingEpisode == null ? 0 : startingEpisode); i < (episodes == null ? 500 : episodes); i++) {
            PlayResult<GameAction, GameStateImage> result = game.play();
            saveAnimatedGif(result.getHistory().stream()
                    .map(HistoryEntry::getObservation)
                    .map(GameStateImage::getObservation)
                    .toList(), i);
            greedy.decrease(i);
            if (saveModel) {
                network.save(filename);
            }
            log.info("Round: {}, Reward: {}, Epsilon: {}, Frames: {}", i, result.getReward(), String.format("%.2f", greedy.getEpsilon()), result.getRounds());
        }
    }

    private void saveAnimatedGif(List<BufferedImage> result, int episode) {
        try (ImageOutputStream output = new FileImageOutputStream(new File("movie" + episode + ".gif"))) {
            GifSequenceWriter gifSequenceWriter = new GifSequenceWriter(output, result.get(0).getType(),1 , false);
            for (BufferedImage buffered : result) {
                gifSequenceWriter.writeToSequence(buffered);
            }
            gifSequenceWriter.close();
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
