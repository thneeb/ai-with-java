package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class Transition<A extends Enum<A>, O extends Observation> {
    private O observation;
    private A action;
    private double reward;
    private O nextObservation;
}
