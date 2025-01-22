package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.Map;

@Getter
@Setter
@Builder
@ToString
public class MultiPlayerResult<A extends Action, O extends Observation> {
    private Map<Agent<A, O>, Double> rewards;
    private O observation;
    private int rounds;
}
