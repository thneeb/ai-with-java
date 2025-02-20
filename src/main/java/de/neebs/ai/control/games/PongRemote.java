package de.neebs.ai.control.games;

import de.neebs.ai.control.rl.QNetwork;
import de.neebs.ai.control.rl.remote.NeuralNetworkImageSequence;
import de.neebs.ai.control.rl.remote.RemoteNetworkFacade;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
public class PongRemote {
    private final RemoteNetworkFacade remoteNetworkFacade;

    QNetwork<Pong.GameAction, Pong.GameStateImageSequence> createQNetwork(String filename) {
        if (filename == null) {
            return new NeuralNetworkImageSequence<>(remoteNetworkFacade);
        } else {
            return new NeuralNetworkImageSequence<>(remoteNetworkFacade, filename + ".txt");
        }
    }
}
