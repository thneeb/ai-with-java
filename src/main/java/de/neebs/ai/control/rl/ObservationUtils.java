package de.neebs.ai.control.rl;

import java.lang.reflect.InvocationTargetException;

public class ObservationUtils {
    private static <T extends Observation2D> T reduceDepth(Observation3D state, Class<T> clazz) {
        try {
            T newState = clazz.getDeclaredConstructor().newInstance();
            for (int col = 0; col < state.getObservation().length; col++) {
                for (int row = 0; row < state.getObservation()[col].length; row++) {
                    newState.getObservation()[col][row] = state.getObservation()[col][row][0];
                }
            }
            return newState;
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            throw new IllegalStateException();
        }
    }
}
