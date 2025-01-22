package de.neebs.ai.control.rl;

import lombok.AccessLevel;
import lombok.Getter;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

@Getter(AccessLevel.PACKAGE)
public class DoubleQLearningAgent<A extends Action, O extends Observation1D> extends QLearningAgent<A, O> implements TrainingListener {
    private final NeuralNetwork1D<O> targetNetwork;

    public DoubleQLearningAgent(NeuralNetwork1D<O> neuralNetwork, EpsilonGreedyPolicy policy, double gamma) {
        super(neuralNetwork, policy, gamma);
        neuralNetwork.setListeners(this);
        this.targetNetwork = neuralNetwork.copy();
    }

    @Override
    public void learn(List<Transition<A, O>> transitions) {
        List<NeuralNetwork1D.TrainingData<O>> trainingData = transitions.stream()
                .map(this::transition2TrainingData)
                .toList();
        getNeuralNetwork().train(trainingData);
    }

    private NeuralNetwork1D.TrainingData<O> transition2TrainingData(Transition<A, O> transition) {
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
        return new NeuralNetwork1D.TrainingData<>(transition.getObservation(), qPrevious);
    }

        @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (iteration % 50 == 0) {
            getTargetNetwork().copyParams(getNeuralNetwork());
        }
    }

    @Override
    public void onEpochStart(Model model) {

    }

    @Override
    public void onEpochEnd(Model model) {

    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {

    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {

    }

    @Override
    public void onGradientCalculation(Model model) {

    }

    @Override
    public void onBackwardPass(Model model) {

    }
}
