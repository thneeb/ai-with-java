package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class ConnectFourTest {
    @Test
    void test() {
        ConnectFour connectFour = new ConnectFour();
        connectFour.execute(false, false, null, 100);
    }

    private static class DoNotWinAgent implements Agent<ConnectFour.GameAction, ConnectFour.GameState> {
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

    @Test
    void test2() {
        int episodeCount = 500;
        ConnectFour.Env environment = new ConnectFour.Env(ConnectFour.GameAction.class, ConnectFour.GameState.class);
        NeuralNetwork1D<ConnectFour.GameState> network = new NeuralNetwork1D<>(new ConnectFour.MyNeuralNetworkFactory(), 1234L);
        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(1.0).epsilonMin(0.01).decreaseRate(0.010).step(1).build();
        Agent<ConnectFour.GameAction, ConnectFour.GameState> red = new QLearningAgent<>(network, greedy, 1.0);
//        Agent<Action, GameState> red = new DoubleQLearningAgent<>(network, greedy, 0.99);
        Agent<ConnectFour.GameAction, ConnectFour.GameState> yellow = new DoNotWinAgent(new ConnectFour.ActionObservationFilter());
        // Agent<Action, GameState> yellow = new RandomAgent();
        MultiPlayerGame<ConnectFour.GameAction, ConnectFour.GameState, ConnectFour.Env> connectFour = new MultiPlayerGame<>(environment, Arrays.asList(yellow, red));
        for (int i = 1; i <= episodeCount; i++) {
            MultiPlayerResult<ConnectFour.GameAction, ConnectFour.GameState> multiPlayerResult = connectFour.play();
            System.out.println("Episode " + i + ", QLearning: " + multiPlayerResult.getRewards().get(red) + ", DoNotWin: " + multiPlayerResult.getRewards().get(yellow) + ", Rounds: " + multiPlayerResult.getRounds() + ", Epsilon: " + greedy.getEpsilon());
            greedy.decrease();
        }
        network.save("connect-four-agent.zip");
    }

    @Test
    void test3() {
        ConnectFour.Env environment = new ConnectFour.Env(ConnectFour.GameAction.class, ConnectFour.GameState.class);
        NeuralNetwork1D<ConnectFour.GameState> network = new NeuralNetwork1D<>("connect-four-agent.zip");
        Agent<ConnectFour.GameAction, ConnectFour.GameState> red = new QLearningAgent<>(network, EpsilonGreedyPolicy.builder().epsilon(0.01).epsilonMin(0.01).decreaseRate(0.010).step(1).build(), 1.0);
        Agent<ConnectFour.GameAction, ConnectFour.GameState> yellow = new DoTheFollowingAgent<>(
                new ConnectFour.ActionObservationFilter(),
                List.of(ConnectFour.GameAction.DROP_3, ConnectFour.GameAction.DROP_2, ConnectFour.GameAction.DROP_1, ConnectFour.GameAction.DROP_6));
        MultiPlayerGame<ConnectFour.GameAction, ConnectFour.GameState, ConnectFour.Env> connectFour = new MultiPlayerGame<>(environment, Arrays.asList(yellow, red));
        for (int i = 1; i <= 100; i++) {
            MultiPlayerResult<ConnectFour.GameAction, ConnectFour.GameState> multiPlayerResult = connectFour.play();
            System.out.println("Episode: " + i + ", QLearning: " + multiPlayerResult.getRewards().get(red) + ", FixAgent: " + multiPlayerResult.getRewards().get(yellow) + ", Rounds: " + multiPlayerResult.getRounds());
        }
        network.save("connect-four-agent.zip");
    }
}
