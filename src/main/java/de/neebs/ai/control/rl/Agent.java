package de.neebs.ai.control.rl;

public interface Agent<A extends Action, O extends Observation> {
    A chooseAction(O observation, ActionSpace<A> actionSpace);
}
