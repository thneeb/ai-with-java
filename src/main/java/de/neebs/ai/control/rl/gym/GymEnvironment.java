package de.neebs.ai.control.rl.gym;

import de.neebs.ai.control.rl.AbstractEnvironment;
import de.neebs.ai.control.rl.Environment;
import de.neebs.ai.control.rl.Observation;
import de.neebs.ai.control.rl.StepResult;

import java.util.List;

public class GymEnvironment<A extends Enum<A>, O extends Observation> extends AbstractEnvironment<A, O> {
    private final GymClient gymClient;
    private String instanceId;

    public GymEnvironment(Class<A> actions, Class<O> observation, GymClient gymClient) {
        super(actions, observation);
        this.gymClient = gymClient;
    }

    public Environment<A, O> init(String envId) {
        instanceId = gymClient.makeEnv(envId);
        return this;
    }

    @Override
    public O reset() {
        return gymClient.reset(instanceId, getObservationClass());
    }

    @Override
    public StepResult<O> step(A action) {
        GymStepResult<O> stepResult = gymClient.step(instanceId, action.ordinal(), getObservationClass());
        return StepResult.<O>builder().done(stepResult.isDone() || stepResult.isTruncated()).reward(stepResult.getReward()).observation(stepResult.getObservation()).build();
    }

    @Override
    public List<Integer> getObservationSpace() {
        return gymClient.getObservationSpace(instanceId);
    }
}
