package de.neebs.ai.control.games;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.neebs.ai.control.rl.*;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

class ConnectFourTest {
    @Test
    void test() {
        ConnectFour connectFour = new ConnectFour();
        connectFour.execute(false, false, null, 100);
    }

    @Test
    void test2() {
        int episodeCount = 500;
        ConnectFour.Env environment = new ConnectFour.Env(ConnectFour.GameAction.class, ConnectFour.GameState.class);
        NeuralNetwork1D<ConnectFour.GameState> network = new NeuralNetwork1D<>(new ConnectFour.NeuralNetworkFactory1D(), 1234L);
        EpsilonGreedyPolicy greedy = EpsilonGreedyPolicy.builder().epsilon(1.0).epsilonMin(0.01).decreaseRate(0.010).step(1).build();
        Agent<ConnectFour.GameAction, ConnectFour.GameState> red = new QLearningAgent<>(network, greedy, 1.0);
//        Agent<Action, GameState> red = new DoubleQLearningAgent<>(network, greedy, 0.99);
        Agent<ConnectFour.GameAction, ConnectFour.GameState> yellow = new ConnectFour.DoNotWinAgent(new ConnectFour.ActionObservationFilter());
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
        String filename = "connect-four-agent-ql.json";
//        NeuralNetwork1D<ConnectFour.GameState> network = new NeuralNetwork1D<>("connect-four-agent.zip");
        SimpleQNetwork<ConnectFour.GameState, ConnectFour.GameAction> network = loadSimpleQNetwork(filename);
        Agent<ConnectFour.GameAction, ConnectFour.GameState> red = new QLearningAgent<>(network, EpsilonGreedyPolicy.builder().epsilon(0.01).epsilonMin(0.01).decreaseRate(0.010).step(1).build(), 1.0);
        Agent<ConnectFour.GameAction, ConnectFour.GameState> yellow = new DoTheFollowingAgent<>(
                new ConnectFour.ActionObservationFilter(),
                List.of(ConnectFour.GameAction.DROP_3, ConnectFour.GameAction.DROP_3, ConnectFour.GameAction.DROP_4, ConnectFour.GameAction.DROP_6, ConnectFour.GameAction.DROP_5));
        MultiPlayerGame<ConnectFour.GameAction, ConnectFour.GameState, ConnectFour.Env> connectFour = new MultiPlayerGame<>(environment, Arrays.asList(red, yellow));
        for (int i = 1; i <= 100; i++) {
            MultiPlayerResult<ConnectFour.GameAction, ConnectFour.GameState> multiPlayerResult = connectFour.play();
            System.out.println("Episode: " + i + ", QLearning: " + multiPlayerResult.getRewards().get(red) + ", FixAgent: " + multiPlayerResult.getRewards().get(yellow) + ", Rounds: " + multiPlayerResult.getRounds());
        }
        network.save(filename);
    }

    private static SimpleQNetwork<ConnectFour.GameState, ConnectFour.GameAction> loadSimpleQNetwork(String filename) {
        SimpleQNetwork<ConnectFour.GameState, ConnectFour.GameAction> network;
        File file = new File(filename);
        try {
            List<SimpleQNetwork.SaveData<ConnectFour.GameState>> saveData = new ObjectMapper().readValue(file, new TypeReference<>() {});
            network = new SimpleQNetwork<>(saveData.stream().collect(Collectors.toMap(SimpleQNetwork.SaveData::getObservation, SimpleQNetwork.SaveData::getWeights)), 0.001, ConnectFour.GameAction.class);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return network;
    }

}
