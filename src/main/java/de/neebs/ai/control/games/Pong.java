package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.Observation;
import de.neebs.ai.control.rl.gym.GymClient;
import de.neebs.ai.control.rl.gym.GymStepResult;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

@Service
@RequiredArgsConstructor
public class Pong {
    private final GymClient gymClient;

    @Getter
    @Setter
    public static class GameState implements Observation {
        private int[][][] observation;

        private int getWidth() {
            return observation.length;
        }

        private int getHeight() {
            return observation[0].length;
        }

        @Override
        public double[] getFlattenedObservations() {
            return new double[0];
        }
    }

    private static class Utils {
        private Utils() {
        }

        private static void saveImage(GameState state, String name) {
            BufferedImage image = new BufferedImage(state.getWidth(), state.getHeight(), BufferedImage.TYPE_INT_RGB );
            for (int col = 0; col < state.getWidth(); col++) {
                for (int row = 0; row < state.getHeight(); row++) {
                    int rgb = (state.observation[col][row][0] << 16) | (state.observation[col][row][1] << 8) | state.observation[col][row][2];
                    image.setRGB(col, row, rgb);
                }
            }
            try {
                File outputFile = new File(name);
                ImageIO.write(image, "bmp", outputFile);
            } catch (IOException e) {
                throw new IllegalStateException();
            }
        }
    }

    public void execute() {
        String instanceId = gymClient.makeEnv("ale_py:ALE/Pong-v5");
        GameState state = gymClient.reset(instanceId, GameState.class);

        boolean done = false;
        int i = 0;
        while (!done) {
            String imageName = "output-" + String.format("%05d", i) + ".bmp";
            Utils.saveImage(state, imageName);
            GymStepResult<GameState> stepResult = gymClient.step(instanceId, 0, GameState.class);
            state = stepResult.getObservation();
            done = stepResult.isDone();
            i++;
        }
    }
}
