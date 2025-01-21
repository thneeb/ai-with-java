package de.neebs.ai.control.rl;

public class NextFreeAgent<A extends Enum<A>, O extends Observation> implements Agent<A, O>{
    private final ActionFilter<A, O> actionFilter;

    public NextFreeAgent(ActionFilter<A, O> actionFilter) {
        this.actionFilter = actionFilter;
    }

    @Override
    public A chooseAction(O observation, ActionSpace<A> actionSpace) {
        return actionFilter.filter(observation, actionSpace).getActions().get(0);
    }
}
