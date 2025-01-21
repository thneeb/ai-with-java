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
public class PlayResult<O extends Observation> {
    private double reward;
    private O observation;
    private int rounds;
}
