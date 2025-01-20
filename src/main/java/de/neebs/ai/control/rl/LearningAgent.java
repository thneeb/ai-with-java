package de.neebs.ai.control.rl;

import java.util.List;

public interface LearningAgent<A extends Enum<A>, O extends Observation> extends Agent<A, O> {
    void learn(List<Transition<A, O>> transitions);
}
