package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.List;
import java.util.Map;

@Getter
@Setter
@Builder
@ToString
public class PlayResult<A extends Action, O extends Observation> {
    private double reward;
    private O observation;
    private int rounds;
    private List<HistoryEntry<A, O>> history;
}
