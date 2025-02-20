package de.neebs.ai.control.rl;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class TrainingData<O extends Observation> {
    private O observation;
    private double[] output;
    private int action;
}
