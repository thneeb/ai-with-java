package de.neebs.ai.control.rl;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.*;

@RequiredArgsConstructor
@Getter(AccessLevel.PACKAGE)
public class QLearningAgent<A extends Action, O extends Observation> implements LearningAgent<A, O> {
    private final QNetwork<O> neuralNetwork;
    private final EpsilonGreedyPolicy policy;
    private final double gamma;

    @Override
    public A chooseAction(O observation, ActionSpace<A> actionSpace) {
        if (policy.isExploration()) {
            return actionSpace.getRandomAction();
        } else {
            double[] q = getNeuralNetwork().predict(observation);
            List<Double> iList = new ArrayList<>(Arrays.stream(q).boxed().toList());
            double max = Collections.max(iList);
            int i = iList.indexOf(max);
            return actionSpace.getActions().get(i);
        }
    }

    @Override
    public void learn(List<Transition<A, O>> transitions) {
        List<TrainingData<O>> trainingData = transitions.stream()
                .map(this::transition2TrainingData)
                .toList();
        getNeuralNetwork().train(trainingData);
    }

    private TrainingData<O> transition2TrainingData(Transition<A, O> transition) {
        double[] qValues = getNeuralNetwork().predict(transition.getObservation());

        makeQValuesPositive(qValues);

        double q;
        if (transition.getNextObservation() == null) {
            q = 0.0;
        } else {
            double[] qNext = getNeuralNetwork().predict(transition.getNextObservation());
            makeQValuesPositive(qNext);
            q = Arrays.stream(qNext).max().orElse(0.0);
        }
        double target = transition.getReward() + q * getGamma();
        qValues[transition.getAction().ordinal()] = target;
        return new TrainingData<>(transition.getObservation(), qValues, transition.getAction().ordinal());
    }

    void makeQValuesPositive(double[] qValues) {
        double maxQ = Arrays.stream(qValues).max().orElse(0);

        if (maxQ < 0) {
            double minQ = Arrays.stream(qValues).min().orElse(0);
            for (int i = 0; i < qValues.length; i++) {
                qValues[i] += Math.abs(minQ) + 0.01;  // Kleinen Offset basierend auf negativstem Wert
            }
        }
    }
}
