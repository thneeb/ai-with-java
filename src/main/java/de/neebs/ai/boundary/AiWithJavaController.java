package de.neebs.ai.boundary;

import de.neebs.ai.control.games.ConnectFour;
import de.neebs.ai.control.dl4j.FirstExample;
import de.neebs.ai.control.games.Pong;
import de.neebs.ai.control.network.NetworkMain;
import de.neebs.ai.control.games.TicTacToe;
import de.neebs.ai.control.rl.gym.GymClient;
import de.neebs.aiwithjava.client.boundary.DefaultApi;
import de.neebs.ai.control.perceptron.PerceptronMain;
import de.neebs.aiwithjava.client.entity.InstanceId;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class AiWithJavaController implements DefaultApi {
    private final PerceptronMain perceptronMain;
    private final NetworkMain networkMain;
    private final FirstExample firstExample;
    private final TicTacToe ticTacToe;
    private final ConnectFour connectFour;
    private final Pong pong;
    private final GymClient gymClient;

    @Override
    public ResponseEntity<Void> startNetwork() {
        networkMain.execute();
        return ResponseEntity.ok().build();
    }

    @Override
    public ResponseEntity<Void> startPerceptron() {
        perceptronMain.execute();
        return ResponseEntity.ok().build();
    }

    @Override
    public ResponseEntity<Void> startDL4J() {
        firstExample.execute();
        return ResponseEntity.ok().build();
    }

    @Override
    public ResponseEntity<Void> ticTacToe() {
        ticTacToe.execute();
        return ResponseEntity.ok().build();
    }

    @Override
    public ResponseEntity<Void> connectFour() {
        connectFour.execute();
        return ResponseEntity.ok().build();
    }

    @Override
    public ResponseEntity<Void> pong() {
        pong.execute();
        return ResponseEntity.ok().build();
    }

    @Override
    public ResponseEntity<InstanceId> createGym() {
        return ResponseEntity.ok(InstanceId.builder().instanceId(gymClient.makeEnv("ale_py:ALE/Pong-v5")).build());
    }

    @Override
    public ResponseEntity<Object> resetGym(String instanceId) {
        return ResponseEntity.ok(gymClient.reset(instanceId, Pong.GameState.class));
    }
}
