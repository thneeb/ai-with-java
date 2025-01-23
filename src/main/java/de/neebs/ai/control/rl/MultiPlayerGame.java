package de.neebs.ai.control.rl;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

@Slf4j
@Getter
public class MultiPlayerGame<A extends Action, O extends MultiPlayerState & Observation1D, E extends AbstractEnvironment<A, O>> {
    private final E environment;
    private final List<Agent<A, O>> agents;
    private final ReplayBuffer<A, O> replayBuffer = new ReplayBuffer<>(1000);

    public MultiPlayerGame(E environment, List<Agent<A, O>> agents) {
        this.environment = environment;
        this.agents = agents;
    }

    public Agent<A, O> nextAgent() {
        return getAgents().get(getEnvironment().getCurrentObservation().getPlayer());
    }

    public MultiPlayerResult<A, O> play() {
        Map<Agent<A, O>, Double> rewards = new HashMap<>();

        List<HistoryEntry<A, O>> history = new ArrayList<>();
        O observation = getEnvironment().reset();
        boolean done = false;
        while (!done) {
            Agent<A, O>agent = nextAgent();
            A action = agent.chooseAction(observation, getEnvironment().getActionSpace());
            StepResult<O> stepResult = getEnvironment().step(action);
            history.add(new HistoryEntry<>(agent, observation, action, stepResult.getReward()));
            double reward = rewards.getOrDefault(agent, 0d);
            reward += stepResult.getReward();
            rewards.put(agent, reward);
            done = stepResult.isDone();
            observation = stepResult.getObservation();
        }

        Collections.reverse(history);
        for (Agent<A, O> a : agents) {
            if (a instanceof LearningAgent<A, O> learningAgent) {
                List<HistoryEntry<A, O>> ownMoves = history.stream().filter(f -> f.getAgent().equals(a)).toList();
                for (int i = 0; i < ownMoves.size(); i++) {
                    HistoryEntry<A, O> entry = ownMoves.get(i);
                    double reward = entry.getReward();
                    if (i == 0) {
                        HistoryEntry<A, O> last = history.get(0);
                        if (!last.getAgent().equals(a)) {
                            reward = -last.getReward();
                        }
                    }
                    replayBuffer.add(Transition.<A, O>builder()
                            .observation(entry.getObservation())
                            .action(entry.getAction())
                            .reward(reward)
                            .nextObservation(i == 0 ? null : ownMoves.get(i - 1).getObservation())
                            .build());
                }
            }
        }

        for (Agent<A, O> a : agents) {
            if (a instanceof LearningAgent<A, O> learningAgent) {
                List<Transition<A, O>> transitions = replayBuffer.sample(history.size());
                learningAgent.learn(transitions);
            }
        }

        return MultiPlayerResult.<A, O>builder()
                .rewards(rewards)
                .rounds(history.size())
                .observation(observation)
                .build();
    }
}
