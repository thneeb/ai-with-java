package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;

@Builder
@Getter
public class HistoryEntry<A extends Enum<A>, O extends Observation> {
    private final Agent<A, O> agent;
    private final O observation;
    private final A action;
    private double reward;
}
