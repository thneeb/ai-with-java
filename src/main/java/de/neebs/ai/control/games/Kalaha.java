package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import lombok.Getter;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class Kalaha {
    enum GameAction implements Action {
        A1, A2, A3, A4, A5, A6
    }

    @Getter
    public static class GameState extends MultiPlayerState implements Observation1D{
        private int[][] board;

        public GameState() {
            super(2);
            init();
        }

        private void init() {
            board = new int[2][6];
            for (int i = 0; i < 6; i++) {
                board[0][i] = 4;
                board[1][i] = 4;
            }
        }

        public int[] getBoard(int player) {
            return board[player];
        }

        @Override
        public double[] getFlattenedObservation() {
            return DoubleStream.concat(IntStream.of(board[0]).mapToDouble(f -> (double) f), IntStream.of(board[1]).mapToDouble(f -> (double) f)).toArray();
        }

        public GameState copy() {
            GameState copy = new GameState();
            for (int i = 0; i < 6; i++) {
                copy.board[0][i] = board[0][i];
                copy.board[1][i] = board[1][i];
            }
            return copy;
        }
    }

    static class ActionObservationFilter implements ActionFilter<GameAction, GameState> {
        @Override
        public ActionSpace<GameAction> filter(GameState observation, ActionSpace<GameAction> actions) {
            List<GameAction> as = Arrays.stream(GameAction.values()).filter(f -> observation.getBoard(observation.getPlayer())[f.ordinal()] != 0).toList();
            return new ActionSpace<>(as);
        }
    }

    static class Env extends AbstractEnvironment<GameAction, GameState> {
        public Env(Class<GameAction> actions, Class<GameState> observation) {
            super(actions, observation);
        }

        @Override
        public List<Integer> getObservationSpace() {
            return getShape(getCurrentObservation().board);
        }

        @Override
        public StepResult<GameState> step(GameAction action) {
            GameState gameState = getCurrentObservation().copy();
            int player = gameState.getPlayer();
            int[][] board = gameState.getBoard();
            int stones = board[player][action.ordinal()];
            board[player][action.ordinal()] = 0;
            int i = action.ordinal();
            int reward = 0;
            while (stones > 0) {
                i++;
                if (i == 6) {
                    reward++;
                } else {
                    if (i == 7) {
                        i = 0;
                        player = 1 - player;
                    }
                    board[player][i]++;
                }
                stones--;
            }
            if (player == gameState.getPlayer() && i != 6 && board[player][i] == 1 && board[1 - player][5 - i] > 0) {
                reward += board[1 - player][5 - i];
                board[1 - player][5 - i] = 0;
                reward += board[player][i];
                board[player][i] = 0;
            }
            if (i != 6) {
                gameState.nextPlayer();
            }
            boolean done = false;
            if (Arrays.stream(board[player]).allMatch(f -> f == 0)) {
                for (int j = 0; j < 6; j++) {
                    reward += board[1 - player][j];
                    board[1 - player][j] = 0;
                }
                done = true;
            }
            setCurrentObservation(gameState);
            return StepResult.<GameState>builder()
                    .reward(reward)
                    .observation(gameState)
                    .done(done)
                    .build();
        }
    }

    public void execute() {
    }
}
