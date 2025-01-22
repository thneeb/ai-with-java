package de.neebs.ai.control.rl;

import lombok.Getter;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Getter
public class ActionSpace<T extends Action> {
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

    public T getRandomAction() {
        return actions.get(RANDOM.nextInt(actions.size()));
    }
}
