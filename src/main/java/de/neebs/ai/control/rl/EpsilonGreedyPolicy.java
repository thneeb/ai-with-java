package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;

import java.util.Random;

@Builder
public class EpsilonGreedyPolicy {
    private static final Random RANDOM = new Random();
    @Getter
    private double epsilon;
    private final double epsilonMin;
    private final double decreaseRate;
    @Getter
    private final int step;

    public EpsilonGreedyPolicy(double epsilon, double epsilonMin, double decreaseRate, int step) {
        this.epsilon = epsilon;
        this.epsilonMin = epsilonMin;
        this.decreaseRate = decreaseRate;
        this.step = (step == 0 ? 1 : step);
    }

    public void decrease(int step) {
        if (step % this.step == 0) {
            decrease();
        }
    }

    public boolean isExploration() {
        return RANDOM.nextDouble() < getEpsilon();
    }

    public void decrease() {
        if (epsilon > epsilonMin) {
            epsilon -= decreaseRate;
        }
    }
}
