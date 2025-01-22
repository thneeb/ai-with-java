package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.Agent;
import de.neebs.ai.control.rl.MultiPlayerGame;
import de.neebs.ai.control.rl.RandomAgent;
import org.junit.jupiter.api.Test;

import java.util.List;

class KalahaTest {
    @Test
    void testManualGame() {
        Kalaha.ActionObservationFilter filter = new Kalaha.ActionObservationFilter();
        Agent<Kalaha.GameAction, Kalaha.GameState> agent1 = new RandomAgent<>(filter);
        Agent<Kalaha.GameAction, Kalaha.GameState> agent2 = new RandomAgent<>(filter);
        Kalaha.Env env = new Kalaha.Env(Kalaha.GameAction.class, Kalaha.GameState.class);
        MultiPlayerGame<Kalaha.GameAction, Kalaha.GameState, Kalaha.Env> kalaha = new MultiPlayerGame<>(env, List.of(agent1, agent2));
        kalaha.play();
    }
}
