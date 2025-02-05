package de.neebs.ai.boundary;

import de.neebs.ai.control.games.ConnectFour;
import de.neebs.ai.control.dl4j.FirstExample;
import de.neebs.ai.control.games.Pong;
import de.neebs.ai.control.network.NetworkMain;
import de.neebs.ai.control.games.TicTacToe;
import de.neebs.ai.control.rl.StepResult;
import de.neebs.ai.control.rl.gym.GymClient;
import de.neebs.aiwithjava.client.boundary.DefaultApi;
import de.neebs.ai.control.perceptron.PerceptronMain;
import de.neebs.aiwithjava.client.entity.Config;
import de.neebs.aiwithjava.client.entity.InstanceId;
import de.neebs.aiwithjava.client.entity.Observation2D;
import de.neebs.aiwithjava.client.entity.ResetConnectFourRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;

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
    public ResponseEntity<Void> connectFour(Config config) {
        connectFour.execute(config.getStartFresh(), config.getSaveModel(), config.getSaveInterval(), config.getEpisodes());
        return ResponseEntity.ok().build();
    }

    @Override
    public ResponseEntity<Observation2D> resetConnectFour(ResetConnectFourRequest resetConnectFourRequest) {
        ConnectFour.GameState gameState = connectFour.reset(resetConnectFourRequest.getStarter().getValue());
        Observation2D obs = Observation2D.builder()
                .board(convertConnectFourBoard(gameState))
                .py(gameState.getPlayer())
                .done(false)
                .build();
        return ResponseEntity.ok(obs);
    }

    @Override
    public ResponseEntity<Observation2D> stepConnectFour(Integer actionId, Observation2D requestBody) {
        StepResult<ConnectFour.GameState> result = connectFour.step(requestBody.getBoard(), actionId, requestBody.getPy());
        Observation2D obs = convert(result);
        return ResponseEntity.ok(obs);
    }

    private Observation2D convert(StepResult<ConnectFour.GameState> result) {
        String winner = result.getReward() < 0 ? "HUMAN" : result.getReward() > 0 ? "AI" : null;
        return Observation2D.builder()
                .board(convertConnectFourBoard(result.getObservation()))
                .py(result.getObservation().getPlayer())
                .done(result.isDone())
                .winner(winner)
                .build();
    }

    @Override
    public ResponseEntity<Void> pong(Config config) {
        pong.execute(config.getStartFresh(), config.getSaveModel(), config.getEpsilon(), config.getEpisodes());
        return ResponseEntity.ok().build();
    }

    @Override
    public ResponseEntity<InstanceId> createGym() {
        return ResponseEntity.ok(InstanceId.builder().instanceId(gymClient.makeEnv("ale_py:ALE/Pong-v5")).build());
    }

    @Override
    public ResponseEntity<Object> resetGym(String instanceId) {
        return ResponseEntity.ok(gymClient.reset(instanceId, Pong.GameState3D.class));
    }

    private List<List<Integer>> convertConnectFourBoard(ConnectFour.GameState state) {
        List<List<Integer>> result = new ArrayList<>();
        for (double[] row: state.getBoard()) {
            List<Integer> r = new ArrayList<>();
            for (double cell: row) {
                r.add((int) cell);
            }
            result.add(r);
        }
        return result;
    }
}
