package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.*;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

class TicTacToeNetworkTest {
    @Test
    void testLearning() {
        TicTacToe.GameState gameState = new TicTacToe.GameState();
        gameState.setPlayer(1);
        gameState.getBoard()[TicTacToe.GameAction.MIDDLE_MIDDLE.ordinal()] = 1;
        NeuralNetwork1D<TicTacToe.GameState> network = new NeuralNetwork1D<>(new TicTacToe.MyNeuralNetworkFactory(), new Random().nextLong());
        for (TicTacToe.GameAction a : TicTacToe.GameAction.values()) {
            double[] prediction = network.predict(gameState);
            prediction[a.ordinal()] = gameState.getFlattenedObservation()[a.ordinal()] == 0 ? 0 : -1;
            network.train(new NeuralNetwork1D.TrainingData<>(gameState, prediction));
        }
        System.out.println(Arrays.toString(network.predict(gameState)));
    }

    @Test
    void testOneGame() {
        NeuralNetwork1D<TicTacToe.GameState> network = new NeuralNetwork1D<>(new TicTacToe.MyNeuralNetworkFactory(), new Random().nextLong());
        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(0).epsilonMin(0).decreaseRate(0.001).step(1).build();
        Agent<TicTacToe.GameAction, TicTacToe.GameState> oAgent = new QLearningAgent<>(network, greedy, 0.99);
        Agent<TicTacToe.GameAction, TicTacToe.GameState> xAgent = new NextFreeAgent<>(new TicTacToe.ActionObservationFilter());
        TicTacToe.Env env = new TicTacToe.Env(TicTacToe.GameAction.class, TicTacToe.GameState.class);
        MultiPlayerGame<TicTacToe.GameAction, TicTacToe.GameState, TicTacToe.Env> ticTacToe = new MultiPlayerGame<>(env, List.of(xAgent, oAgent));
        MultiPlayerResult<TicTacToe.GameAction, TicTacToe.GameState> multiPlayerResult = ticTacToe.play();
        System.out.println(multiPlayerResult);
    }
}
