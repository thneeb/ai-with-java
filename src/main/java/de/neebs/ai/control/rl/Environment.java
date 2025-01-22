package de.neebs.ai.control.rl;

import java.util.List;

public interface Environment<A extends Action, O extends Observation> {
    O reset();

    StepResult<O> step(A action);

    ActionSpace<A> getActionSpace();

    Class<O> getObservationClass();

    List<Integer> getObservationSpace();

    O getCurrentObservation();
}
