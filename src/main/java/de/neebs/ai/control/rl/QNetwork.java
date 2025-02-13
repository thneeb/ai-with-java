package de.neebs.ai.control.rl;

import java.util.List;

public interface QNetwork<O extends Observation> {
    double[] predict(O observation);

    void train(O observation, double[] target);

    void train(List<TrainingData<O>> trainingData);

    void save(String filename);

    void copyParams(QNetwork<O> source);

    QNetwork<O> copy();
}
