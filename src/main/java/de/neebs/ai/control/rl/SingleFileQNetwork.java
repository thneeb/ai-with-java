package de.neebs.ai.control.rl;

import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class SingleFileQNetwork<O extends Observation, A extends Action> implements QNetwork<O> {
    private static final Random RANDOM = new Random();
    private static final ObjectMapper mapper = new ObjectMapper();
    private final Map<O, List<Double>> qValues;
    private final double learnRate;
    private final Class<A> actionClass;
    @Getter
    private int miss = 0;
    @Getter
    private int hit = 0;

    public SingleFileQNetwork(double learnRate, Class<A> actionClass) {
        qValues = new HashMap<>();
        this.learnRate = learnRate;
        this.actionClass = actionClass;
    }

    public SingleFileQNetwork(String filename, Class<O> clazz, double learnRate, Class<A> actionClass) {
        try {
            JavaType type = mapper.getTypeFactory().constructParametricType(SaveQWeights.class, clazz);
            JavaType type2 = mapper.getTypeFactory().constructParametricType(ArrayList.class, type);
            File file = new File(filename);
            List<SaveQWeights<O>> data = mapper.readValue(file, type2);
            qValues = data.stream().collect(Collectors.toMap(SaveQWeights::getObservation, SaveQWeights::getWeights));
            this.learnRate = learnRate;
            this.actionClass = actionClass;
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void train(O observation, double[] target) {
        List<Double> list = initQList(observation);
        for (int i = 0; i < target.length; i++) {
            list.set(i, list.get(i) + (target[i] - list.get(i)) * learnRate);
        }
    }

    @Override
    public void train(List<TrainingData<O>> trainingData) {
        for (TrainingData<O> data : trainingData) {
            train(data.getInput(), data.getOutput());
        }
    }

    @Override
    public double[] predict(O observation) {
        List<Double> list = initQList(observation);
        return list.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private List<Double> initQList(O observation) {
        List<Double> list = qValues.get(observation);
        if (list == null) {
            list = new ArrayList<>();
            for (A action : actionClass.getEnumConstants()) {
                list.add(RANDOM.nextDouble(1));
            }
            qValues.put(observation, list);
            miss++;
        } else {
            hit++;
        }
        return list;
    }

    @Override
    public void save(String filename) {
        try {
            new ObjectMapper().writeValue(new File(filename), qValues.entrySet().stream().map(f -> new SaveQWeights<>(f.getKey(), f.getValue())).toList());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public QNetwork<O> copy() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copyParams(QNetwork<O> other) {
        throw new UnsupportedOperationException();
    }

    public int getSize() {
        return qValues.size();
    }
}
