package de.neebs.ai.control.rl;

import lombok.Builder;
import lombok.Getter;

@Builder
public class EpsilonGreedyPolicy {
    @Getter
    private double epsilon;
    private final double epsilonMin;
    private final double decreaseRate;
    @Getter
    private final int step;

    public EpsilonGreedyPolicy(double epsilonStart, double epsilonMin, double decreaseRate, int step) {
        this.epsilon = epsilonStart;
        this.epsilonMin = epsilonMin;
        this.decreaseRate = decreaseRate;
        this.step = step;
    }

    public double getEpsilon(int step) {
        if (step % this.step == 0) {
            epsilon = Math.max(epsilonMin, epsilon - decreaseRate);
        }
        return epsilon;
    }

    public double decrementEpsilon(int step) {
        if (step % this.step == 0) {
            epsilon = Math.max(epsilonMin, epsilon - decreaseRate);
        }
        return epsilon;
    }

    public double decrementEpsilon() {
        epsilon = Math.max(epsilonMin, epsilon - decreaseRate);
        return epsilon;
    }
}
