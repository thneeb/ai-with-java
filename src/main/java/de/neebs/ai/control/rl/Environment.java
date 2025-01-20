package de.neebs.ai.control.rl;

import lombok.Getter;
import lombok.Setter;

import java.lang.reflect.InvocationTargetException;

@Getter
public abstract class Environment<A extends Enum<A>, O extends Observation> {
    private final ActionSpace<A> actionSpace;
    private final Class<O> observationClass;
    @Setter
    private O currentObservation;

    protected Environment(Class<A> actions, Class<O> observation) {
        try {
            actionSpace = new ActionSpace<>(actions);
            observationClass = observation;
            currentObservation = observation.getDeclaredConstructor().newInstance();
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            throw new IllegalArgumentException(e);
        }
    }

    public O reset() {
        try {
            currentObservation = observationClass.getDeclaredConstructor().newInstance();
            return currentObservation;
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            throw new IllegalArgumentException(e);
        }
    }

    public abstract ActionSpace<A> getActionSpaceForObservation(O observation);

    public ActionSpace<A> getActionSpaceForObservation() {
        return getActionSpaceForObservation(currentObservation);
    }

    public abstract StepResult<O> step(A action);

}
