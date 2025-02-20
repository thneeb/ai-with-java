package de.neebs.ai.control.rl;

import lombok.Getter;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.List;

@Getter
public abstract class ObservationWrapper<A extends Action, Oin extends Observation, Oout extends Observation> implements Environment<A, Oout> {
    private final Environment<A, Oin> environment;

    public ObservationWrapper(Environment<A, Oin> environment) {
        this.environment = environment;
    }

    protected abstract Oout wrapper(Oin observation, boolean initialize);

    @Override
    public abstract List<Integer> getObservationSpace();

    @Override
    public final Oout reset() {
        return wrapper(environment.reset(), true);
    }

    @Override
    public final StepResult<Oout> step(A action) {
        StepResult<Oin> stepResult = environment.step(action);
        return new StepResult<>(wrapper(stepResult.getObservation(), false), stepResult.getReward(), stepResult.isDone());
    }

    @Override
    public final ActionSpace<A> getActionSpace() {
        return environment.getActionSpace();
    }

    @Override
    @SuppressWarnings("unchecked")
    public final Class<Oout> getObservationClass() {
        try {
            Type type = ((ParameterizedType)getClass().getGenericSuperclass()).getActualTypeArguments()[2];
            return (Class<Oout>)Class.forName(type.getTypeName());
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public final Oout getCurrentObservation() {
        return wrapper(environment.getCurrentObservation(), false);
    }
}
