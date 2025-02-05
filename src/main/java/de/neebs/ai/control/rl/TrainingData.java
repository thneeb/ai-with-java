package de.neebs.ai.control.rl;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class TrainingData<O extends Observation> {
    private O input;
    private double[] output;
    private int index;
}
