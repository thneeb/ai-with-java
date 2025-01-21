package de.neebs.ai.control.rl.gym;

import de.neebs.ai.control.rl.*;
import lombok.Getter;

import java.util.List;

@Getter
public abstract class ObservationWrapper<A extends Enum<A>, Oin extends Observation, Oout extends Observation> implements Environment<A, Oout> {
    private final Environment<A, Oin> environment;

    public ObservationWrapper(Environment<A, Oin> environment) {
        this.environment = environment;
    }

    protected abstract Oout wrapper(Oin observation);

    @Override
    public abstract List<Integer> getObservationSpace();

    @Override
    public final Oout reset() {
        return wrapper(environment.reset());
    }

    @Override
    public final StepResult<Oout> step(A action) {
        StepResult<Oin> stepResult = environment.step(action);
        return new StepResult<>(wrapper(stepResult.getObservation()), stepResult.getReward(), stepResult.isDone());
    }

    @Override
    public final ActionSpace<A> getActionSpace() {
        return environment.getActionSpace();
    }

    @Override
    public final Class<Oout> getObservationClass() {
        return null;
    }

    @Override
    public final Oout getCurrentObservation() {
        return wrapper(environment.getCurrentObservation());
    }
}
