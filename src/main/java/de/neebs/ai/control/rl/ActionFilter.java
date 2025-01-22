package de.neebs.ai.control.rl;

public interface ActionFilter<A extends Action, O extends Observation> {
    ActionSpace<A> filter(O observation, ActionSpace<A> actions);
}
