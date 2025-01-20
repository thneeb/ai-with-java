package de.neebs.ai.control.rl;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;

@RequiredArgsConstructor
public abstract class MultiPlayerObservation implements Observation {
    private final int playerCount;

    @Getter
    @Setter
    private int player;

    public void nextPlayer() {
        player = (player + 1) % playerCount;
    }
}
