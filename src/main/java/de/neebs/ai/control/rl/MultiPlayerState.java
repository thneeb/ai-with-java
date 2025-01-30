package de.neebs.ai.control.rl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;

@RequiredArgsConstructor
@EqualsAndHashCode
public abstract class MultiPlayerState {
    private final int playerCount;

    @Getter
    @Setter
    private int player;

    public void nextPlayer() {
        player = (player + 1) % playerCount;
    }
}
