package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class TicTacToeNetworkTest {
    @Test
    void test() {
        NeuralNetwork network = new NeuralNetwork(new TicTacToe.MyNeuralNetworkFactory());
        double[] input = {1, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        for (int i = 0; i < 10; i++) {
            double[] output = network.predict(input);
            System.out.println(Arrays.toString(output));
            output[0] = -1;
            network.train(NeuralNetwork.TrainingData.builder().input(input).output(output).build());
        }
    }

    @Test
    void testLearning() {
        TicTacToe.GameState gameState = new TicTacToe.GameState();
        gameState.setPlayer(1);
        gameState.getBoard()[TicTacToe.GameAction.MIDDLE_MIDDLE.ordinal()] = 1;
        NeuralNetwork network = new NeuralNetwork(new TicTacToe.MyNeuralNetworkFactory());
        int episodes = 1000;
        for (int i = 0; i < episodes; i++) {
            for (TicTacToe.GameAction a : TicTacToe.GameAction.values()) {
                double[] prediction = network.predict(gameState.getFlattenedObservations());
                prediction[a.ordinal()] = gameState.getFlattenedObservations()[a.ordinal()] == 0 ? 0 : -10;
                network.train(NeuralNetwork.TrainingData.builder().input(gameState.getFlattenedObservations()).output(prediction).build());
            }
        }
        System.out.println(Arrays.toString(network.predict(gameState.getFlattenedObservations())));
    }

    @Test
    void testOneGame() {
        NeuralNetwork network = new NeuralNetwork(new TicTacToe.MyNeuralNetworkFactory());
        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(0).epsilonMin(0).decreaseRate(0.001).step(1).build();
        Agent<TicTacToe.GameAction, TicTacToe.GameState> oAgent = new QLearningAgent<>(network, greedy, 0.99); // TicTacToe.RandomAgent();
        Agent<TicTacToe.GameAction, TicTacToe.GameState> xAgent = new NextFreeAgent<>();
        TicTacToe.Env env = new TicTacToe.Env(TicTacToe.GameAction.class, TicTacToe.GameState.class);
        MultiPlayerGame<TicTacToe.GameAction, TicTacToe.GameState, TicTacToe.Env> ticTacToe = new MultiPlayerGame<>(env, List.of(xAgent, oAgent));
        MultiPlayerResult<TicTacToe.GameAction, TicTacToe.GameState> multiPlayerResult = ticTacToe.play();
        System.out.println(multiPlayerResult);
    }
}
