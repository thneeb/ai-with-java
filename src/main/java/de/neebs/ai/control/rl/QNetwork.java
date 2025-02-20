package de.neebs.ai.control.rl;

import java.util.List;

public interface QNetwork<A extends Action, O extends Observation> {
    double[] predict(O observation);

    void train(O observation, double[] target);

    void train(List<TrainingData<O>> trainingData);

    default boolean isFastTrainingSupported() {
        return false;
    }

    default void train(List<Transition<A, O>> transitions, double gamma) {
        throw new UnsupportedOperationException("Fast training is not supported");
    }

    void save(String filename);

    void copyParams(QNetwork<A, O> source);

    QNetwork<A, O> copy();
}
