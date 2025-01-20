package de.neebs.ai.control.rl.gym;

import de.neebs.ai.control.rl.Observation;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.Map;

@Getter
@Setter
@Builder
public class GymStepResult<O extends Observation> {
    private O observation;
    private double reward;
    private boolean done;
    private boolean truncated;
    private Map<String, Object> info;
}
