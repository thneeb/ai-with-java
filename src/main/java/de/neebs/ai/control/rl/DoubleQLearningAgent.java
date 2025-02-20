package de.neebs.ai.control.rl;

import java.util.Arrays;
import java.util.List;

public class DoubleQLearningAgent<A extends Action, O extends Observation> extends QLearningAgent<A, O> {
    private final QNetwork<A, O> targetNetwork;
    private final int updateFrequency;
    private int updateCounter = 0;

    public DoubleQLearningAgent(QNetwork<A, O> neuralNetwork, EpsilonGreedyPolicy policy, double gamma, int updateFrequency) {
        super(neuralNetwork, policy, gamma);
        this.targetNetwork = neuralNetwork.copy();
        this.updateFrequency = updateFrequency;
    }

    @Override
    public void learn(List<Transition<A, O>> transitions) {
        if (getNeuralNetwork().isFastTrainingSupported()) {
            getNeuralNetwork().train(transitions, getGamma());
        } else {
            List<TrainingData<O>> trainingData = transitions.stream()
                    .map(this::transition2TrainingData)
                    .toList();
            getNeuralNetwork().train(trainingData);
        }
        updateCounter++;
        if (updateCounter >= updateFrequency) {
            updateCounter -= updateFrequency;
            targetNetwork.copyParams(getNeuralNetwork());
        }
    }

    private TrainingData<O> transition2TrainingData(Transition<A, O> transition) {
        double[] qPrevious = getNeuralNetwork().predict(transition.getObservation());
        double q;
        if (transition.isDone()) {
            q = 0.0;
        } else {
            double[] qNext = targetNetwork.predict(transition.getNextObservation());
            q = Arrays.stream(qNext).max().orElse(0.0);
        }
        double target = transition.getReward() + q * getGamma();
        qPrevious[transition.getAction().ordinal()] = target;
        return new TrainingData<>(transition.getObservation(), qPrevious, transition.getAction().ordinal());
    }
}
