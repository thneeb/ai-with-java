package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import de.neebs.ai.control.rl.dl4j.NeuralNetwork1D;
import de.neebs.ai.control.rl.dl4j.NeuralNetworkFactory;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.DoubleStream;

@Slf4j
@Service
public class TicTacToe  {
    public enum GameAction implements Action {
        UPPER_LEFT, UPPER_MIDDLE, UPPER_RIGHT, MIDDLE_LEFT, MIDDLE_MIDDLE, MIDDLE_RIGHT, LOWER_LEFT, LOWER_MIDDLE, LOWER_RIGHT
    }

    static class ActionObservationFilter implements ActionFilter<GameAction, GameState> {
        @Override
        public ActionSpace<GameAction> filter(GameState observation, ActionSpace<GameAction> actions) {
            List<GameAction> filteredActions = new ArrayList<>();
            for (GameAction action : actions.getActions()) {
                if (observation.getBoard()[action.ordinal()] == 0) {
                    filteredActions.add(action);
                }
            }
            return new ActionSpace<>(filteredActions);
        }
    }

    public static class Env extends AbstractEnvironment<GameAction, GameState> {
        public Env(Class<GameAction> actions, Class<GameState> observation) {
            super(actions, observation);
        }

        @Override
        public List<Integer> getObservationSpace() {
            return getShape(getCurrentObservation().board);
        }

        @Override
        public StepResult<GameState> step(GameAction action) {
            GameState newState = getCurrentObservation().copy();
            if (newState.getBoard()[action.ordinal()] != 0) {
                return StepResult.<GameState>builder().observation(newState).done(true).reward(-1).build();
            }
            newState.getBoard()[action.ordinal()] = getCurrentObservation().getPlayer() + 1;
            newState.setPlayer((getCurrentObservation().getPlayer() + 1) % 2);
            setCurrentObservation(newState);
            boolean gameWon = TicTacToeUtils.checkWin(newState);
            boolean remis = !gameWon && TicTacToeUtils.checkRemis(newState);
            return StepResult.<GameState>builder()
                    .observation(newState)
                    .done(gameWon || remis)
                    .reward(gameWon ? 1 : 0)
                    .build();
        }
    }

    @Getter
    @ToString
    public static class GameState extends MultiPlayerState implements Observation1D {
        private double[] board = new double[9];

        public GameState() {
            super(2);
            init();
        }

        private void init() {
            board = new double[] {
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0};
            setPlayer(0);
        }

        @Override
        public double[] getFlattenedObservation() {
            return DoubleStream.concat(Arrays.stream(board), DoubleStream.of(getPlayer())).toArray();
        }

        public GameState copy()  {
            GameState gameState = new GameState();
            gameState.board = Arrays.copyOf(board, board.length);
            gameState.setPlayer(getPlayer());
            return gameState;
        }
    }

    static class MyTicTacToeAgent implements Agent<GameAction, GameState> {
        private final ActionObservationFilter filterActions;

        MyTicTacToeAgent() {
            filterActions = new ActionObservationFilter();
        }

        @Override
        public GameAction chooseAction(GameState observation, ActionSpace<GameAction> actionSpace) {
            actionSpace = filterActions.filter(observation, actionSpace);
            for (GameAction action : actionSpace.getActions()) {
                GameState gameState = observation.copy();
                // check if we can win
                gameState.getBoard()[action.ordinal()] = observation.getPlayer() + 1;
                if (TicTacToeUtils.checkWin(gameState)) {
                    return action;
                }
                // check if we can prevent a win
                gameState.getBoard()[action.ordinal()] = (observation.getPlayer() + 1) % 2 + 1;
                if (TicTacToeUtils.checkWin(gameState)) {
                    return action;
                }
            }
            if (observation.getBoard()[GameAction.MIDDLE_MIDDLE.ordinal()] == 0) {
                return GameAction.MIDDLE_MIDDLE;
            }
            if (observation.getBoard()[GameAction.UPPER_MIDDLE.ordinal()] != 0 && observation.getBoard()[GameAction.UPPER_MIDDLE.ordinal()] == observation.getBoard()[GameAction.MIDDLE_LEFT.ordinal()]) {
                return GameAction.UPPER_LEFT;
            }
            if (observation.getBoard()[GameAction.UPPER_MIDDLE.ordinal()] != 0 && observation.getBoard()[GameAction.UPPER_MIDDLE.ordinal()] == observation.getBoard()[GameAction.MIDDLE_RIGHT.ordinal()]) {
                return GameAction.UPPER_RIGHT;
            }
            if (observation.getBoard()[GameAction.LOWER_MIDDLE.ordinal()] != 0 && observation.getBoard()[GameAction.LOWER_MIDDLE.ordinal()] == observation.getBoard()[GameAction.MIDDLE_LEFT.ordinal()]) {
                return GameAction.LOWER_LEFT;
            }
            if (observation.getBoard()[GameAction.LOWER_MIDDLE.ordinal()] != 0 && observation.getBoard()[GameAction.LOWER_MIDDLE.ordinal()] == observation.getBoard()[GameAction.MIDDLE_RIGHT.ordinal()]) {
                return GameAction.LOWER_RIGHT;
            }
            return actionSpace.getRandomAction();
        }
    }

    private static class TicTacToeUtils {
        private TicTacToeUtils() {
        }

        private static boolean checkRemis(GameState gameState) {
            return gameState.board[0] != 0 && gameState.board[1] != 0 && gameState.board[2] != 0
                    && gameState.board[3] != 0 && gameState.board[4] != 0 && gameState.board[5] != 0
                    && gameState.board[6] != 0 && gameState.board[7] != 0 && gameState.board[8] != 0;
        }

        private static boolean checkWin(GameState gameState) {
            if (gameState.board[0] == gameState.board[1] && gameState.board[1] == gameState.board[2] && gameState.board[0] != 0)
                return true;
            if (gameState.board[3] == gameState.board[4] && gameState.board[4] == gameState.board[5] && gameState.board[3] != 0)
                return true;
            if (gameState.board[6] == gameState.board[7] && gameState.board[7] == gameState.board[8] && gameState.board[6] != 0)
                return true;
            if (gameState.board[0] == gameState.board[3] && gameState.board[3] == gameState.board[6] && gameState.board[0] != 0)
                return true;
            if (gameState.board[1] == gameState.board[4] && gameState.board[4] == gameState.board[7] && gameState.board[1] != 0)
                return true;
            if (gameState.board[2] == gameState.board[5] && gameState.board[5] == gameState.board[8] && gameState.board[2] != 0)
                return true;
            if (gameState.board[0] == gameState.board[4] && gameState.board[4] == gameState.board[8] && gameState.board[0] != 0)
                return true;
            if (gameState.board[2] == gameState.board[4] && gameState.board[4] == gameState.board[6] && gameState.board[2] != 0)
                return true;
            return false;
        }
    }

    public static class MyNeuralNetworkFactory implements NeuralNetworkFactory {
        @Override
        public MultiLayerNetwork createNeuralNetwork(long seed) {
            int input = new GameState().getFlattenedObservation().length;
            int output = GameAction.values().length;
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Sgd(0.01))
                    .miniBatch(false)
//                .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(input)
                            .nOut(input * 2)
                            .activation(Activation.SIGMOID)
//                        .weightInit(new UniformDistribution(0, 1))
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nOut(input * 2)
                            .activation(Activation.SIGMOID)
//                        .weightInit(new UniformDistribution(0, 1))
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .nOut(output)
                            .activation(Activation.RELU)
//                        .weightInit(new UniformDistribution(0, 1))
                            .build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            return net;
        }
    }

    public void execute() {
        int episodeCount = 300;
        NeuralNetwork1D<GameState> network = new NeuralNetwork1D<>(new MyNeuralNetworkFactory(), new Random().nextLong());
        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(0.01).epsilonMin(0.01).decreaseRate(0.001).step(1).build();
//        Agent<GameAction, GameState> oAgent = new QLearningAgent1D<>(network, greedy, 0.99);
        Agent<GameAction, GameState> oAgent = new DoubleQLearningAgent<>(network, greedy, 0.99, 100);
        Agent<GameAction, GameState> xAgent = new MyTicTacToeAgent();
//        Agent<Action, GameState> xAgent = new RandomAgent();
//        Agent<Action, GameState> xAgent = new NextFreeAgent();
        Env env = new Env(GameAction.class, GameState.class);
        MultiPlayerGame<GameAction, GameState, Env> ticTacToe = new MultiPlayerGame<>(env, List.of(xAgent, oAgent));
        Map<Agent<GameAction, GameState>, Integer> result = new HashMap<>();
        for (int i = 1; i <= episodeCount; i++) {
            MultiPlayerResult<GameAction, GameState> multiPlayerResult = ticTacToe.play();
            greedy.decrease(i);

            for (Map.Entry<Agent<GameAction, GameState>, Double> entry : multiPlayerResult.getRewards().entrySet()) {
                result.put(entry.getKey(), result.getOrDefault(entry.getKey(), 0) + (entry.getValue() == 1 ? 1 : 0));
            }

            if (i % 100 == 0) {
                log.info("Episode: {}, {}, {}, {}", i, result.get(xAgent), result.get(oAgent), greedy.getEpsilon());
                result = new HashMap<>();
            }
        }

        for (int i = 0; i < 50; i++) {
            MultiPlayerResult<GameAction, GameState> multiPlayerResult = ticTacToe.play();
            Optional<Map.Entry<Agent<GameAction, GameState>, Double>> optional =
                    multiPlayerResult.getRewards().entrySet().stream().filter(f -> f.getValue() == 1).findAny();

            if (optional.isPresent()) {
                log.info("Winner: {}", optional.get());
            } else {
                log.info("Remis");
            }
        }
        log.info("Epsilon: {}", greedy.getEpsilon());
    }
}
