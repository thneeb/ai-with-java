package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.MultiPlayerGame;
import org.junit.jupiter.api.Test;

import java.util.List;

class KalahaTest {
    @Test
    void testManualGame() {
        Kalaha.Env env = new Kalaha.Env(Kalaha.Action.class, Kalaha.GameState.class);
        MultiPlayerGame<Kalaha.Action, Kalaha.GameState, Kalaha.Env> kalaha = new MultiPlayerGame<>(env, List.of());
        kalaha.play();
    }
}
