package de.neebs.ai.control.rl;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.*;

@RequiredArgsConstructor
@Getter(AccessLevel.PACKAGE)
public class QLearningAgent<A extends Enum<A>, O extends Observation> implements LearningAgent<A, O> {
    private final NeuralNetwork neuralNetwork;
    private final EpsilonGreedyPolicy policy;
    private final double gamma;
    private static final Random RANDOM = new Random();

    @Override
    public A chooseAction(O observation, ActionSpace<A> actionSpace) {
        if (RANDOM.nextDouble() < policy.getEpsilon()) {
            return actionSpace.getRandomAction();
        } else {
            double[] q = neuralNetwork.predict(observation.getFlattenedObservations());
            List<Double> iList = new ArrayList<>(Arrays.stream(q).boxed().toList());
            int i = iList.indexOf(Collections.max(iList));
            return actionSpace.getAllActions().get(i);
        }
    }

    @Override
    public void learn(List<Transition<A, O>> transitions) {
        List<NeuralNetwork.TrainingData> trainingData = transitions.stream()
                .map(this::transition2TrainingData)
                .toList();
        getNeuralNetwork().train(trainingData);
    }

    private NeuralNetwork.TrainingData transition2TrainingData(Transition<A, O> transition) {
        double[] qPrevious = getNeuralNetwork().predict(transition.getObservation().getFlattenedObservations());
        double q;
        if (transition.getNextObservation() == null) {
            q = 0.0;
        } else {
            double[] qNext = getNeuralNetwork().predict(transition.getNextObservation().getFlattenedObservations());
            q = Arrays.stream(qNext).max().orElse(0.0);
        }
        double target = transition.getReward() + q * getGamma();
        qPrevious[transition.getAction().ordinal()] = target;
        return new NeuralNetwork.TrainingData(transition.getObservation().getFlattenedObservations(), qPrevious);
    }
}
