package de.neebs.ai.control.rl;

import lombok.AccessLevel;
import lombok.Getter;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

@Getter(AccessLevel.PACKAGE)
public class DoubleQLearningAgent<A extends Action, O extends Observation1D> extends QLearningAgent<A, O> {
    private final QNetwork<O> targetNetwork;
    private final int updateFrequency;
    private int updateCounter = 0;

    public DoubleQLearningAgent(QNetwork<O> neuralNetwork, EpsilonGreedyPolicy policy, double gamma, int updateFrequency) {
        super(neuralNetwork, policy, gamma);
        this.targetNetwork = neuralNetwork.copy();
        this.updateFrequency = updateFrequency;
    }

    @Override
    public void learn(List<Transition<A, O>> transitions) {
        List<TrainingData<O>> trainingData = transitions.stream()
                .map(this::transition2TrainingData)
                .toList();
        getNeuralNetwork().train(trainingData);
        updateCounter += trainingData.size();
        if (updateCounter >= updateFrequency) {
            updateCounter -= updateFrequency;
            targetNetwork.copyParams(getNeuralNetwork());
        }
    }

    private TrainingData<O> transition2TrainingData(Transition<A, O> transition) {
        double[] qPrevious = getNeuralNetwork().predict(transition.getObservation());
        double q;
        if (transition.getNextObservation() == null) {
            q = 0.0;
        } else {
            double[] qNext = getNeuralNetwork().predict(transition.getNextObservation());
            double qTemp = Arrays.stream(qNext).max().orElse(0.0);
            int index = IntStream.range(0, qNext.length).filter(i -> qNext[i] == qTemp).findFirst().orElseThrow();
            double[] qTarget = getTargetNetwork().predict(transition.getNextObservation());
            q = qTarget[index];
        }
        double target = transition.getReward() + q * getGamma();
        qPrevious[transition.getAction().ordinal()] = target;
        return new TrainingData<>(transition.getObservation(), qPrevious, transition.getAction().ordinal());
    }
}
