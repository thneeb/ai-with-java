package de.neebs.ai.control.rl;

import lombok.Getter;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Getter
public class ActionSpace<T extends Enum<T>> {
    private static final Random RANDOM = new Random();

    private final List<T> actions;

    public ActionSpace(T[] actions) {
        this.actions = Arrays.asList(actions);
    }

    public ActionSpace(List<T> actions) {
        this.actions = actions;
    }

    public ActionSpace(Class<T> actionClass) {
        this.actions = List.of(actionClass.getEnumConstants());
    }

    public List<T> getAllActions() {
        return List.of(actions.get(0).getDeclaringClass().getEnumConstants());
    }

    public boolean contains(T action) {
        return actions.contains(action);
    }

    public T getRandomAction() {
        return actions.get(RANDOM.nextInt(actions.size()));
    }

    public int ordinal(T action) {
        return getAllActions().indexOf(action);
    }
}
