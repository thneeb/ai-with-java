package de.neebs.ai.control.rl;

import lombok.AccessLevel;
import lombok.Getter;

import java.util.List;

@Getter(value = AccessLevel.PROTECTED)
public abstract class EnvironmentWrapper<A extends Enum<A>, O extends Observation> implements Environment<A, O> {
    private final Environment<A, O> environment;

    public EnvironmentWrapper(Environment<A, O> environment) {
        this.environment = environment;
    }

    @Override
    public O reset() {
        return environment.reset();
    }

    @Override
    public StepResult<O> step(A action) {
        return environment.step(action);
    }

    @Override
    public ActionSpace<A> getActionSpace() {
        return environment.getActionSpace();
    }

    @Override
    public Class<O> getObservationClass() {
        return environment.getObservationClass();
    }

    @Override
    public O getCurrentObservation() {
        return environment.getCurrentObservation();
    }

    @Override
    public List<Integer> getObservationSpace() {
        return environment.getObservationSpace();
    }
}
