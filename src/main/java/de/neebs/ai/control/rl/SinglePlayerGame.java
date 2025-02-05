package de.neebs.ai.control.rl;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

@Slf4j
@Getter
public class SinglePlayerGame<A extends Action, O extends Observation, E extends Environment<A, O>> {
    private final E environment;
    private final Agent<A, O> agent;
    private final ReplayBuffer<A, O> replayBuffer;
    private final int batchSize;
    private final double rewardPercentage;

    public SinglePlayerGame(E environment, Agent<A, O> agent, int bufferSize, int batchSize, double rewardPercentage) {
        this.environment = environment;
        this.agent = agent;
        this.replayBuffer = new ReplayBuffer<>(bufferSize);
        this.batchSize = batchSize;
        this.rewardPercentage = rewardPercentage;
    }

    public PlayResult<A, O> play() {
        List<HistoryEntry<A, O>> history = new ArrayList<>();
        O observation = getEnvironment().reset();
        double reward = 0.0;
        boolean done = false;
        while (!done) {
            A action = agent.chooseAction(observation, getEnvironment().getActionSpace());
            StepResult<O> stepResult = getEnvironment().step(action);
            history.add(new HistoryEntry<>(agent, observation, action, stepResult.getReward()));
            reward += stepResult.getReward();
            done = stepResult.isDone();
            replayBuffer.add(Transition.<A, O>builder()
                    .observation(observation)
                    .action(action)
                    .reward(stepResult.getReward())
                    .nextObservation(done ? null : stepResult.getObservation())
                    .build());

            if (agent instanceof LearningAgent<A, O> learningAgent && replayBuffer.size() >= batchSize) {
                List<Transition<A, O>> transitions = replayBuffer.sample(batchSize, rewardPercentage);
                learningAgent.learn(transitions);
            }

            observation = stepResult.getObservation();
        }

        return PlayResult.<A, O>builder()
                .reward(reward)
                .rounds(history.size())
                .history(history)
                .observation(observation)
                .build();
    }
}
