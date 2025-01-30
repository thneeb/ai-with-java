package de.neebs.ai.control.rl;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class SaveQWeights<O extends Observation> {
    private O observation;
    private List<Double> weights;
}
