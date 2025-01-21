package de.neebs.ai.control.rl;

import lombok.AllArgsConstructor;
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

    public EpsilonGreedyPolicy(double epsilon, double epsilonMin, double decreaseRate, int step) {
        this.epsilon = epsilon;
        this.epsilonMin = epsilonMin;
        this.decreaseRate = decreaseRate;
        this.step = (step == 0 ? 1 : step);
    }

    public void decrementEpsilon(int step) {
        if (step % this.step == 0) {
            decrementEpsilon();
        }
    }

    public void decrementEpsilon() {
        epsilon = Math.max(epsilonMin, epsilon - decreaseRate);
    }
}
