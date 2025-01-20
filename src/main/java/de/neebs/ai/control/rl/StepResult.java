package de.neebs.ai.control.rl;

import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StepResult<O extends Observation> {
    private O observation;
    private double reward;
    private boolean done;
}
