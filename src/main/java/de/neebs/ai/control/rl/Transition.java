package de.neebs.ai.control.rl;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Transition<A extends Action, O extends Observation> {
    private O observation;
    private A action;
    private double reward;
    private O nextObservation;
}
