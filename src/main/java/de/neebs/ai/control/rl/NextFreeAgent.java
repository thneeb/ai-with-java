package de.neebs.ai.control.rl;

public class NextFreeAgent<A extends Enum<A>, O extends Observation> implements Agent<A, O>{
    @Override
    public A chooseAction(O observation, ActionSpace<A> actionSpace) {
        return actionSpace.getActions().get(0);
    }
}
