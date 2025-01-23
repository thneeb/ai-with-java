package de.neebs.ai.control.rl;

import java.util.List;

public class DoTheFollowingAgent<A extends Action, O extends Observation> implements Agent<A, O> {
    private final ActionFilter<A, O> actionObservationFilter;
    private final List<A> actions;
    private int actionCounter = 0;

    public DoTheFollowingAgent(ActionFilter<A, O> actionObservationFilter, List<A> actions) {
        this.actionObservationFilter = actionObservationFilter;
        this.actions = actions;
    }

    @Override
    public A chooseAction(O observation, ActionSpace<A> actionSpace) {
        actionSpace = actionObservationFilter.filter(observation, actionSpace);
        actionCounter = actionCounter % actions.size();
        A ga = actions.get(actionCounter);
        if (actionSpace.getActions().contains(ga)) {
            actionCounter++;
            return ga;
        } else {
            return actionSpace.getRandomAction();
        }
    }
}
