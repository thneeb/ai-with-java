package de.neebs.ai.control.rl;

import java.util.List;

public abstract class RewardFitter<A extends Action, O extends Observation> implements Environment<A, O> {
    private final Environment<A, O> environment;

    public RewardFitter(Environment<A, O> environment) {
        this.environment = environment;
    }

    protected abstract double fitReward(double reward);

    @Override
    public O reset() {
        return environment.reset();
    }

    @Override
    public StepResult<O> step(A action) {
        StepResult<O> stepResult = environment.step(action);
        return new StepResult<>(stepResult.getObservation(), fitReward(stepResult.getReward()), stepResult.isDone());
    }

    @Override
    public ActionSpace<A> getActionSpace() {
        return environment.getActionSpace();
    }

    @Override
    public List<Integer> getObservationSpace() {
        return environment.getObservationSpace();
    }

    @Override
    public Class<O> getObservationClass() {
        return environment.getObservationClass();
    }

    @Override
    public O getCurrentObservation() {
        return environment.getCurrentObservation();
    }
}
