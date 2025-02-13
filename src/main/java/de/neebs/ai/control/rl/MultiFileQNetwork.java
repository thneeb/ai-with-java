package de.neebs.ai.control.rl;

import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import org.springframework.util.DigestUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.text.MessageFormat;
import java.util.*;
import java.util.stream.Collectors;

public class MultiFileQNetwork<O extends Observation, A extends Action> implements QNetwork<O> {
    private static final Random RANDOM = new Random();
    private static final ObjectMapper mapper = new ObjectMapper();
    private final double learnRate;
    private final Class<A> actionClass;
    private final String filePattern = "d:/myqs/{0}.json";
    @Getter
    private int miss = 0;
    @Getter
    private int hit = 0;

    public MultiFileQNetwork(double learnRate, Class<A> actionClass) {
        this.learnRate = learnRate;
        this.actionClass = actionClass;
    }

    @Override
    public void train(O observation, double[] target) {
        Map<O, List<Double>> map = readFile(observation);
        List<Double> list = map.get(observation);
        for (int i = 0; i < target.length; i++) {
            list.set(i, list.get(i) + (target[i] - list.get(i)) * learnRate);
        }
        writeFile(map);
    }

    @Override
    public void train(List<TrainingData<O>> trainingData) {
        for (TrainingData<O> data : trainingData) {
            train(data.getObservation(), data.getOutput());
        }
    }

    @Override
    public double[] predict(O observation) {
        Map<O, List<Double>> map = readFile(observation);
        List<Double> list = map.get(observation);
        return list.stream().mapToDouble(Double::doubleValue).toArray();
    }

    @Override
    public void save(String filename) {
    }

    @Override
    public QNetwork<O> copy() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copyParams(QNetwork<O> source) {
        throw new UnsupportedOperationException();
    }

    private String getUniqueName(O observation) {
        return DigestUtils.md5DigestAsHex(observation.toString().getBytes(StandardCharsets.UTF_8));
    }

    private Map<O, List<Double>> readFile(O observation) {
        try {
            String filename = MessageFormat.format(filePattern, getUniqueName(observation));
            File file = new File(filename);
            if (file.exists()) {
                JavaType type = mapper.getTypeFactory().constructParametricType(SaveQWeights.class, observation.getClass());
                JavaType type2 = mapper.getTypeFactory().constructParametricType(ArrayList.class, type);
                List<SaveQWeights<O>> data = mapper.readValue(file, type2);
                Map<O, List<Double>> qValues = data.stream().collect(Collectors.toMap(SaveQWeights::getObservation, SaveQWeights::getWeights));
                if (qValues.containsKey(observation)) {
                    hit++;
                } else {
                    List<Double> list = new ArrayList<>();
                    for (A action : actionClass.getEnumConstants()) {
                        list.add(RANDOM.nextDouble(1));
                    }
                    qValues.put(observation, list);
                    miss++;
                }
                return qValues;
            } else {
                List<Double> list = new ArrayList<>();
                for (A action : actionClass.getEnumConstants()) {
                    list.add(RANDOM.nextDouble(1));
                }
                miss++;
                return Map.of(observation, list);
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    private void writeFile(Map<O, List<Double>> map) {
        try {
            List<SaveQWeights<O>> data = map.entrySet().stream().map(f -> new SaveQWeights<>(f.getKey(), f.getValue())).toList();
            String filename = MessageFormat.format(filePattern, getUniqueName(map.keySet().iterator().next()));
            mapper.writeValue(new File(filename), data);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
