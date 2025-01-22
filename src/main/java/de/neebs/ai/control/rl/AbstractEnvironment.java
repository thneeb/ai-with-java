package de.neebs.ai.control.rl;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

@Getter
public abstract class AbstractEnvironment<A extends Action, O extends Observation> implements Environment<A, O> {
    private final ActionSpace<A> actionSpace;
    private final Class<O> observationClass;
    @Setter(value = AccessLevel.PROTECTED)
    private O currentObservation;

    protected AbstractEnvironment(Class<A> actions, Class<O> observation) {
        try {
            actionSpace = new ActionSpace<>(actions);
            observationClass = observation;
            currentObservation = observation.getDeclaredConstructor().newInstance();
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            throw new IllegalArgumentException(e);
        }
    }

    protected List<Integer> getShape(Object object) {
        List<Integer> result = new ArrayList<>();
        Class<?> clazz = object.getClass();
        while (clazz.isArray()) {
            Object[] array = (Object[]) object;
            result.add(array.length);
            clazz = clazz.getComponentType();
        }
        return result;
    }

    @Override
    public O reset() {
        try {
            currentObservation = observationClass.getDeclaredConstructor().newInstance();
            return currentObservation;
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            throw new IllegalArgumentException(e);
        }
    }

}
