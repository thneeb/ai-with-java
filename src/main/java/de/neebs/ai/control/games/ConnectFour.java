package de.neebs.ai.control.games;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.neebs.ai.control.rl.*;
import de.neebs.ai.control.rl.dl4j.NeuralNetwork1D;
import de.neebs.ai.control.rl.dl4j.NeuralNetworkFactory;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

@Service
@Slf4j
public class ConnectFour {
    public enum GameAction implements Action {
        DROP_0, DROP_1, DROP_2, DROP_3, DROP_4, DROP_5, DROP_6
    }

    @Getter
    @Setter
    @ToString
    @EqualsAndHashCode(callSuper = true)
    public static class GameState extends MultiPlayerState implements Observation1D, Observation2D {
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
        @JsonIgnore
        public double[][] getObservation() {
            return board;
        }

        @Override
        @JsonIgnore
        public double[] getFlattenedObservation() {
            return DoubleStream.concat(Stream.of(board).flatMapToDouble(Arrays::stream), DoubleStream.of(getPlayer())).toArray();
        }

        public GameState copy() {
            GameState state = new GameState();
            state.setPlayer(getPlayer());
            state.setBoard(Arrays.stream(getBoard()).map(double[]::clone).toArray(double[][]::new));
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
                Utils.step(state, action);
                double reward = Utils.checkWin(state) == (state.getPlayer() + 1) ? 1 : 0;
                state.nextPlayer();
                setCurrentObservation(state);
                return new StepResult<>(getCurrentObservation(), reward, reward != 0 || Utils.checkRemis(getCurrentObservation()));
            } catch (IllegalMoveException e) {
                return new StepResult<>(getCurrentObservation(), -1, true);
            }
        }

        public void setObservation(GameState observation) {
            super.setCurrentObservation(observation);
        }
    }

    static class Utils {
        private Utils() {
        }

        static boolean checkRemis(GameState observation) {
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

        static int checkWin(GameState observation) {
            for (int row = 0; row < 6; row++) {
                for (int col = 0; col < 7; col++) {
                    if (observation.getBoard()[row][col] != 0) {
                        int player = checkWin(observation, row, col);
                        if (player != 0) {
                            return player;
                        }
                    }
                }
            }
            return 0;
        }

        static int checkWin(GameState observation, int row, int col) {
            double player = observation.getBoard()[row][col];
            if (row + 3 < 6 && observation.getBoard()[row + 1][col] == player && observation.getBoard()[row + 2][col] == player && observation.getBoard()[row + 3][col] == player) {
                return (int)player;
            }
            if (col + 3 < 7 && observation.getBoard()[row][col + 1] == player && observation.getBoard()[row][col + 2] == player && observation.getBoard()[row][col + 3] == player) {
                return (int)player;
            }
            if (row + 3 < 6 && col + 3 < 7 && observation.getBoard()[row + 1][col + 1] == player && observation.getBoard()[row + 2][col + 2] == player && observation.getBoard()[row + 3][col + 3] == player) {
                return (int)player;
            }
            if (row - 3 >= 0 && col + 3 < 7 && observation.getBoard()[row - 1][col + 1] == player && observation.getBoard()[row - 2][col + 2] == player && observation.getBoard()[row - 3][col + 3] == player) {
                return (int)player;
            }
            return 0;
        }

        static void step(GameState observation, GameAction action) throws IllegalMoveException {
            if (observation.getBoard()[0][action.ordinal()] != 0) {
                throw new IllegalMoveException();
            }
            for (int row = 5; row >= 0; row--) {
                if (observation.getBoard()[row][action.ordinal()] == 0) {
                    observation.getBoard()[row][action.ordinal()] = observation.getPlayer() + 1;
                    break;
                }
            }
        }
    }

    static class ConnectFourAgent implements Agent<GameAction, GameState> {
        private int player;

        private final ActionFilter<GameAction, GameState> actionFilter;
        private final EpsilonGreedyPolicy policy;
        private final int depth;

        private int[] minimax(GameState state, ActionSpace<GameAction> actionSpace, int depth, int alpha, int beta, boolean maximizingPlayer) {
            try {
                ActionSpace<GameAction> filteredActionSpace = actionFilter.filter(state, actionSpace);
                boolean isTerminal = Utils.checkWin(state) != 0 || Utils.checkRemis(state);

                if (depth == 0 || isTerminal) {
                    if (isTerminal) {
                        if (Utils.checkWin(state) == player + 1) return new int[] { -1, 100000000 };
                        else if (Utils.checkWin(state) != player + 1) return new int[] { -1, -100000000 };
                        else return new int[] { -1, 0 };
                    } else {
                        int score = (int)Arrays.stream(state.getBoard()).map(f -> f[f.length / 2]).filter(f -> f == player + 1).count() * 3;
                        return new int[] { -1, score };
                    }
                }

                int value;
                GameAction bestCol = filteredActionSpace.getRandomAction();

                if (maximizingPlayer) {
                    value = Integer.MIN_VALUE;
                    for (GameAction col : filteredActionSpace.getActions()) {
                        GameState copy = state.copy();
                        Utils.step(copy, col);
                        int newScore = minimax(copy, actionSpace, depth - 1, alpha, beta, false)[1];
                        if (newScore > value) {
                            value = newScore;
                            bestCol = col;
                        }
                        alpha = Math.max(alpha, value);
                        if (alpha >= beta) break;
                    }
                } else {
                    value = Integer.MAX_VALUE;
                    for (GameAction col : filteredActionSpace.getActions()) {
                        GameState copy = state.copy();
                        copy.nextPlayer();
                        Utils.step(copy, col);
                        copy.nextPlayer();
                        int newScore = minimax(copy, actionSpace, depth - 1, alpha, beta, true)[1];
                        if (newScore < value) {
                            value = newScore;
                            bestCol = col;
                        }
                        beta = Math.min(beta, value);
                        if (alpha >= beta) break;
                    }
                }

                return new int[] { bestCol.ordinal(), value };
            } catch (IllegalMoveException e) {
                throw new RuntimeException(e);
            }
        }

        public ConnectFourAgent(ActionFilter<GameAction, GameState> actionFilter, EpsilonGreedyPolicy policy, int depth) {
            this.actionFilter = actionFilter;
            this.policy = policy;
            this.depth = depth;
        }

        @Override
        public GameAction chooseAction(GameState observation, ActionSpace<GameAction> actionSpace) {
            if (policy.isExploration()) {
                return actionFilter.filter(observation, actionSpace).getRandomAction();
            } else {
                player = observation.getPlayer();
                int[] result = minimax(observation, actionSpace, depth, Integer.MIN_VALUE, Integer.MAX_VALUE, true);
                return actionSpace.getActions().get(result[0]);
            }
        }
    }

    static class DoNotWinAgent implements Agent<ConnectFour.GameAction, ConnectFour.GameState> {
        private final ConnectFour.ActionObservationFilter actionObservationFilter;

        DoNotWinAgent(ConnectFour.ActionObservationFilter actionObservationFilter) {
            this.actionObservationFilter = actionObservationFilter;
        }
        @Override
        public ConnectFour.GameAction chooseAction(ConnectFour.GameState observation, ActionSpace<ConnectFour.GameAction> actionSpace) {
            try {
                actionSpace = actionObservationFilter.filter(observation, actionSpace);
                for (ConnectFour.GameAction action : actionSpace.getActions()) {
                    ConnectFour.GameState state = observation.copy();
                    state.nextPlayer();
                    ConnectFour.Utils.step(state, action);
                    if (ConnectFour.Utils.checkWin(state) > 0) {
                        return action;
                    }
                }
                for (ConnectFour.GameAction action : actionSpace.getActions()) {
                    ConnectFour.GameState state = observation.copy();
                    ConnectFour.Utils.step(state, action);
                    if (ConnectFour.Utils.checkWin(state) == 0) {
                        return action;
                    }
                }
                return actionSpace.getRandomAction();
            } catch (IllegalMoveException e) {
                throw new RuntimeException(e);
            }
        }
    }

    static class NeuralNetworkFactory2D implements NeuralNetworkFactory {
        @Override
        public MultiLayerNetwork createNeuralNetwork(long seed) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345) // Reproduzierbarkeit
                    .weightInit(WeightInit.XAVIER) // Initialisierung für stabile Gradienten
                    .updater(new Adam(0.001)) // Adam-Optimizer mit Lernrate 0.001
                    .list()
                    // 1. Convolutional Layer
                    .layer(0, new ConvolutionLayer.Builder()
                            .kernelSize(3, 3) // 3x3 Filter
                            .stride(1, 1) // Schrittweite
                            .nIn(1) // Eingabekanal
                            .nOut(32) // 32 Filter
                            .activation(Activation.LEAKYRELU) // Aktivierungsfunktion
                            .build())
                    .layer(1, new BatchNormalization()) // Batch-Normalisierung
                    // 2. Convolutional Layer
                    .layer(2, new ConvolutionLayer.Builder()
                            .kernelSize(3, 3)
                            .stride(1, 1)
                            .nOut(64) // 64 Filter
                            .activation(Activation.LEAKYRELU)
                            .build())
                    .layer(3, new BatchNormalization())
                    // 3. Fully Connected Layer
                    .layer(4, new DenseLayer.Builder()
                            .nOut(128) // 128 Neuronen
                            .activation(Activation.LEAKYRELU)
                            .dropOut(0.5) // Dropout für Regularisierung
                            .build())
                    // Output Layer
                    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .nOut(7) // 7 mögliche Züge
                            .activation(Activation.IDENTITY) // Wahrscheinlichkeitsverteilung
                            .build())
                    // Eingabetyp definieren: 6x7 Raster mit einem Kanal
                    .setInputType(InputType.convolutional(6, 7, 1))
                    .build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            return model;
        }
    }

    static class NeuralNetworkFactory1D implements NeuralNetworkFactory {
        @Override
        public MultiLayerNetwork createNeuralNetwork(long seed) {
            int input = new GameState().getFlattenedObservation().length;
            int output = GameAction.values().length;
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                    .updater(new Adam(0.001))
                    .updater(RmsProp.builder().learningRate(0.001).build())
                    .weightInit(WeightInit.XAVIER)
                    .seed(seed)
//                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(1.0)
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(input)
                            .nOut(128)
                            .activation(Activation.LEAKYRELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(128)
                            .nOut(128)
                            .activation(Activation.LEAKYRELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(128)
                            .nOut(64)
                            .activation(Activation.LEAKYRELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .nIn(64)
                            .nOut(output)
                            .activation(Activation.IDENTITY)
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
        private final String yellow;
        private final boolean yellowWon;
        private final boolean yellowLoose;
        private final String red;
        private final boolean redWon;
        private final boolean redLoose;
        private final double epsilon;
        private final int rounds;
    }

    public void execute(boolean startFresh, boolean saveModel, Integer saveInterval, Integer episodes) {
        String filename = "connect-four-agent-1d-dl4j.zip";
        Env environment = new Env(GameAction.class, GameState.class);
        NeuralNetwork1D<GameState> network;
        if (startFresh) {
            network = new NeuralNetwork1D<>(new NeuralNetworkFactory1D(), new Random().nextLong());
        } else {
            network = new NeuralNetwork1D<>(filename);
        }
        /*
        NeuralNetwork2D<GameState> network;
        if (startFresh) {
            network = new NeuralNetwork2D<>(new NeuralNetworkFactory2D(), new Random().nextLong());
        } else {
            network = new NeuralNetwork2D<>(filename);
        }
         */
        /*
        SingleFileQNetwork<GameState, GameAction> network;
        if (startFresh) {
            network = new SingleFileQNetwork<>(0.001, GameAction.class);
        } else {
            network = new SingleFileQNetwork<>(filename, GameState.class, 0.001, GameAction.class);
        }
         */
//        MultiFileQNetwork<GameState, GameAction> network = new MultiFileQNetwork<>(0.001, GameAction.class);

        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(1).epsilonMin(0.01).decreaseRate(0.001).step(1).build();
        Agent<GameAction, GameState> red = new QLearningAgent<>(network, greedy, 0.99);
//        Agent<Action, GameState> red = new DoubleQLearningAgent<>(network, greedy, 0.99);
        Agent<GameAction, GameState> yellow = new ConnectFourAgent(new ActionObservationFilter(), greedy, 7);
//        Agent<GameAction, GameState> yellow = new RandomAgent<>(new ActionObservationFilter());
//        Agent<GameAction, GameState> yellow = new DoNotWinAgent(new ActionObservationFilter());
        MultiPlayerGame<GameAction, GameState, Env> connectFour = new MultiPlayerGame<>(environment, Arrays.asList(red, yellow));
        List<LogEntry> result = new ArrayList<>();
        for (int i = 1; i <= episodes; i++) {
            MultiPlayerResult<GameAction, GameState> multiPlayerResult = connectFour.play();
            greedy.decrease(i);

            result.add(LogEntry.builder()
                    .episode(i)
                    .red(red.getClass().getSimpleName())
                    .redWon(multiPlayerResult.getRewards().get(red) == 1)
                    .redLoose(multiPlayerResult.getRewards().get(red) == -1)
                    .yellow(yellow.getClass().getSimpleName())
                    .yellowWon(multiPlayerResult.getRewards().get(yellow) == 1)
                    .yellowLoose(multiPlayerResult.getRewards().get(yellow) == -1)
                    .rounds(multiPlayerResult.getRounds())
                    .build());

            if (i % 100 == 0) {
                log.info("Episode: {}, red ({}): {}, {}, yellow ({}): {}, {}, rounds: {}, epsilon: {}", i,
                        result.get(0).getRed(),
                        result.stream().filter(LogEntry::isRedWon).count(),
                        result.stream().filter(LogEntry::isRedLoose).count(),
                        result.get(0).getYellow(),
                        result.stream().filter(LogEntry::isYellowWon).count(),
                        result.stream().filter(LogEntry::isYellowLoose).count(),
                        result.stream().mapToInt(LogEntry::getRounds).average().orElse(0),
                        greedy.getEpsilon());
                result = new ArrayList<>();
            }
            if (saveInterval != null && i % saveInterval == 0 && saveModel) {
//                log.info("Misses: {}, Hits: {}", network.getMiss(), network.getHit());
                network.save(filename);
                log.info("Model saved to {}", filename);
            }
        }
//        log.info("Misses: {}, Hits: {}, Percentage: {}", network.getMiss(), network.getHit(), (double)network.getMiss() / (network.getMiss() + network.getHit()));
        if (saveModel) {
            network.save(filename);
            log.info("Model saved to {}", filename);
        }
    }

    public GameState reset(String starter) {
        GameState gameState = new GameState();
        if ("COMPUTER".equals(starter)) {
            gameState = computerMove(gameState).getObservation();
        }
        return gameState;
    }

    public StepResult<GameState> step(List<List<Integer>> state, int action, int player) {
        GameState gameState = new GameState();
        gameState.setPlayer(player);
        gameState.setBoard(state.stream().map(l -> l.stream().mapToDouble(i -> i).toArray()).toArray(double[][]::new));
        boolean done = humanMove(gameState, GameAction.values()[action]);
        if (!done) {
            return computerMove(gameState);
        } else {
            return new StepResult<>(gameState, -1, true);
        }
    }

    private StepResult<GameState> computerMove(GameState gameState) {
//        String filename = "connect-four-agent-ql.json";
        String filename = "connect-four-agent-1d-dl4j.zip";
        Env environment = new Env(GameAction.class, GameState.class);
        environment.setObservation(gameState);
        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(0.01).epsilonMin(0.01).decreaseRate(0.001).step(1).build();
        NeuralNetwork1D<GameState> network = new NeuralNetwork1D<>(filename);
//        SingleFileQNetwork<GameState, GameAction> network = new SingleFileQNetwork<>(filename, GameState.class, 0.001, GameAction.class);
//        QLearningAgent<GameAction, GameState> red = new QLearningAgent<>(network, greedy, 0.99);
        Agent<GameAction, GameState> red = new ConnectFourAgent(new ActionObservationFilter(), greedy, 4);
        GameAction gameAction = red.chooseAction(gameState, new ActionSpace<>(GameAction.class));
        StepResult<GameState> result = environment.step(gameAction);
//        red.learn(List.of(Transition.<GameAction, GameState>builder().observation(gameState).action(gameAction).reward(result.getReward()).nextObservation(result.getObservation()).build()));
  //      network.save(filename);
        return result;
    }

    private boolean humanMove(GameState gameState, GameAction action) {
        try {
            Utils.step(gameState, GameAction.values()[action.ordinal()]);
            gameState.nextPlayer();
        } catch (IllegalMoveException e) {
            throw new IllegalStateException(e);
        }
        return Utils.checkWin(gameState) > 0;
    }
}
