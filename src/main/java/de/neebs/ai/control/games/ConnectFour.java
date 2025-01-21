package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

@Service
@Slf4j
public class ConnectFour {
    enum GameAction implements Action {
        DROP_0, DROP_1, DROP_2, DROP_3, DROP_4, DROP_5, DROP_6
    }

    @Getter
    @Setter
    static class GameState extends MultiPlayerObservation {
        private double[][] board;

        public GameState() {
            super(2);
            init();
        }

        private void init() {
            board = new double[6][7];
            for (double[] row: board) {
                Arrays.fill(row, 0.0);
            }
            setPlayer(0);
        }

        @Override
        public double[] getFlattenedObservation() {
            return DoubleStream.concat(Stream.of(board).flatMapToDouble(Arrays::stream), DoubleStream.of(getPlayer())).toArray();
        }

        public GameState copy() {
            GameState state = new GameState();
            state.setPlayer(getPlayer());
            state.setBoard(Arrays.stream(board).map(double[]::clone).toArray(double[][]::new));
            return state;
        }
    }

    static class ActionObservationFilter implements ActionFilter<GameAction, GameState> {
        @Override
        public ActionSpace<GameAction> filter(GameState observation, ActionSpace<GameAction> actions) {
            List<GameAction> filtered = new ArrayList<>();
            for (GameAction action: actions.getActions()) {
                if (observation.getBoard()[0][action.ordinal()] == 0) {
                    filtered.add(action);
                }
            }
            return new ActionSpace<>(filtered);
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
            try {
                GameState state = getCurrentObservation().copy();
                int row = Utils.step(state, action);
                double reward = Utils.checkWin(state) ? 1 : 0;
                state.nextPlayer();
                setCurrentObservation(state);
                return new StepResult<>(getCurrentObservation(), reward, reward != 0 || Utils.checkRemis(getCurrentObservation()));
            } catch (IllegalMoveException e) {
                return new StepResult<>(getCurrentObservation(), -1, true);
            }
        }
    }

    static class Utils {
        private Utils() {
        }

        private static boolean checkRemis(GameState observation) {
            boolean remis = true;
            for (double[] row: observation.getBoard()) {
                for (double cell: row) {
                    if (cell == 0) {
                        remis = false;
                        break;
                    }
                }
            }
            return remis;
        }

        private static boolean checkWin(GameState observation) {
            for (int row = 0; row < 6; row++) {
                for (int col = 0; col < 7; col++) {
                    if (observation.getBoard()[row][col] != 0 && checkWin(observation, row, col)) {
                        return true;
                    }
                }
            }
            return false;
        }

        private static boolean checkWin(GameState observation, int row, int col) {
            double player = observation.getBoard()[row][col];
            if (row + 3 < 6 && observation.getBoard()[row + 1][col] == player && observation.getBoard()[row + 2][col] == player && observation.getBoard()[row + 3][col] == player) {
                return true;
            }
            if (col + 3 < 7 && observation.getBoard()[row][col + 1] == player && observation.getBoard()[row][col + 2] == player && observation.getBoard()[row][col + 3] == player) {
                return true;
            }
            if (row + 3 < 6 && col + 3 < 7 && observation.getBoard()[row + 1][col + 1] == player && observation.getBoard()[row + 2][col + 2] == player && observation.getBoard()[row + 3][col + 3] == player) {
                return true;
            }
            if (row - 3 >= 0 && col + 3 < 7 && observation.getBoard()[row - 1][col + 1] == player && observation.getBoard()[row - 2][col + 2] == player && observation.getBoard()[row - 3][col + 3] == player) {
                return true;
            }
            return false;
        }

        public static int step(GameState observation, GameAction action) throws IllegalMoveException {
            if (observation.getBoard()[0][action.ordinal()] != 0) {
                throw new IllegalMoveException();
            }
            for (int row = 5; row >= 0; row--) {
                if (observation.getBoard()[row][action.ordinal()] == 0) {
                    observation.getBoard()[row][action.ordinal()] = observation.getPlayer() + 1;
                    return row;
                }
            }
            return -1;
        }
    }

    static class MyConnectFourAgent implements Agent<GameAction, GameState> {
        private final ActionFilter<GameAction, GameState> actionFilter;

        MyConnectFourAgent() {
            actionFilter = new ActionObservationFilter();
        }

        @Override
        public GameAction chooseAction(GameState observation, ActionSpace<GameAction> actionSpace) {
            actionSpace = actionFilter.filter(observation, actionSpace);
            // let's see, if the agent can win in his next draw
            for (GameAction action: actionSpace.getActions()) {
                GameState copy = observation.copy();
                try {
                    int row = Utils.step(copy, action);
                    if (Utils.checkWin(copy, row, action.ordinal())) {
                        return action;
                    }
                } catch (IllegalMoveException e) {
                    // ignore
                }
            }
            // let's see, if the opponent can win in his next draw
            for (GameAction action: actionSpace.getActions()) {
                GameState copy = observation.copy();
                copy.nextPlayer();
                try {
                    int row = Utils.step(copy, action);
                    if (Utils.checkWin(copy, row, action.ordinal())) {
                        return action;
                    }
                } catch (IllegalMoveException e) {
                    // ignore
                }
            }
            return actionSpace.getRandomAction();
        }
    }

    private static class MyNeuralNetworkFactory implements NeuralNetworkFactory {
        @Override
        public MultiLayerNetwork createNeuralNetwork() {
            int input = new GameState().getFlattenedObservation().length;
            int output = GameAction.values().length;
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Adam(0.001))
                    .miniBatch(false)
                    .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(input)
                            .nOut(128)
                            .activation(Activation.SIGMOID)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(128)
                            .nOut(64)
                            .activation(Activation.SOFTSIGN)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .nIn(64)
                            .nOut(output)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            return net;
        }
    }

    @Getter
    @Builder
    private static class LogEntry {
        private final int episode;
        private final boolean yellowWon;
        private final boolean yellowLoose;
        private final boolean redWon;
        private final boolean redLoose;
        private final double epsilon;
        private final int rounds;
    }

    public void execute() {
        int episodeCount = 2000;
        Env environment = new Env(GameAction.class, GameState.class);
        NeuralNetwork1D network = new NeuralNetwork1D(new MyNeuralNetworkFactory());
        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(1).epsilonMin(0.01).decreaseRate(0.001).step(1).build();
        Agent<GameAction, GameState> red = new QLearningAgent<>(network, greedy, 0.99);
//        Agent<Action, GameState> red = new DoubleQLearningAgent<>(network, greedy, 0.99);
        Agent<GameAction, GameState> yellow = new MyConnectFourAgent();
        // Agent<Action, GameState> yellow = new RandomAgent();
        MultiPlayerGame<GameAction, GameState, Env> connectFour = new MultiPlayerGame<>(environment, Arrays.asList(yellow, red));
        List<LogEntry> result = new ArrayList<>();
        for (int i = 0; i < episodeCount; i++) {
            MultiPlayerResult<GameAction, GameState> multiPlayerResult = connectFour.play();
            greedy.decrementEpsilon(i);

            result.add(LogEntry.builder()
                    .episode(i)
                    .redWon(multiPlayerResult.getRewards().get(red) == 1)
                    .redLoose(multiPlayerResult.getRewards().get(red) == -1)
                    .yellowWon(multiPlayerResult.getRewards().get(yellow) == 1)
                    .yellowLoose(multiPlayerResult.getRewards().get(yellow) == -1)
                    .rounds(multiPlayerResult.getRounds())
                    .build());

            if (i % 100 == 0) {
                log.info("Episode: {}, red: {}, {}, yellow: {}, {}, rounds: {}, epsilon: {}", i,
                        result.stream().filter(LogEntry::isRedWon).count(),
                        result.stream().filter(LogEntry::isRedLoose).count(),
                        result.stream().filter(LogEntry::isYellowWon).count(),
                        result.stream().filter(LogEntry::isYellowLoose).count(),
                        result.stream().mapToInt(LogEntry::getRounds).average().orElse(0),
                        greedy.getEpsilon());
                result = new ArrayList<>();
            }
        }
    }
}
