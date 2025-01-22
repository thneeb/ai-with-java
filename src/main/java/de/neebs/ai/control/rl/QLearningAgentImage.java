package de.neebs.ai.control.rl;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.*;

@Getter
@RequiredArgsConstructor
public class QLearningAgentImage<A extends Action, O extends ObservationImage> implements LearningAgent<A, O> {
    private final NeuralNetworkImage<O> neuralNetwork;
    private final EpsilonGreedyPolicy policy;
    private final double gamma;
    private static final Random RANDOM = new Random();

    @Override
    public A chooseAction(O observation, ActionSpace<A> actionSpace) {
        if (RANDOM.nextDouble() < policy.getEpsilon()) {
            return actionSpace.getRandomAction();
        } else {
            double[] q = neuralNetwork.predict(observation);
            List<Double> iList = new ArrayList<>(Arrays.stream(q).boxed().toList());
            int i = iList.indexOf(Collections.max(iList));
            return actionSpace.getActions().get(i);
        }
    }

    @Override
    public void learn(List<Transition<A, O>> transitions) {
        List<NeuralNetworkImage.TrainingData<O>> trainingData = transitions.stream()
                .map(this::transition2TrainingData)
                .toList();
        getNeuralNetwork().train(trainingData);
    }

    private NeuralNetworkImage.TrainingData<O> transition2TrainingData(Transition<A, O> transition) {
        double[] qPrevious = getNeuralNetwork().predict(transition.getObservation());
        double q;
        if (transition.getNextObservation() == null) {
            q = 0.0;
        } else {
            double[] qNext = getNeuralNetwork().predict(transition.getNextObservation());
            q = Arrays.stream(qNext).max().orElse(0.0);
        }
        double target = transition.getReward() + q * getGamma();
        qPrevious[transition.getAction().ordinal()] = target;
        return new NeuralNetworkImage.TrainingData<>(transition.getObservation(), qPrevious);
    }
}
