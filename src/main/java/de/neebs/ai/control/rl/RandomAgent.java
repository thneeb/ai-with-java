package de.neebs.ai.control.rl;

public class RandomAgent<A extends Action, O extends Observation> implements Agent<A, O> {
    private final ActionFilter<A, O> actionFilter;

    public RandomAgent(ActionFilter<A, O> actionFilter) {
        this.actionFilter = actionFilter;
    }

    @Override
    public A chooseAction(O observation, ActionSpace<A> actionSpace) {
        return actionFilter.filter(observation, actionSpace).getRandomAction();
    }
}
