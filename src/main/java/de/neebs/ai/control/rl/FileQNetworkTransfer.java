package de.neebs.ai.control.rl;

import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class FileQNetworkTransfer<A extends Action, O extends Observation> {
    private final ObjectMapper mapper = new ObjectMapper();

    public void splitFile(String filename, Class<O> clazz, Class<A> actionClass) {
        try {
            JavaType type = mapper.getTypeFactory().constructParametricType(SaveQWeights.class, clazz);
            JavaType type2 = mapper.getTypeFactory().constructParametricType(ArrayList.class, type);
            File file = new File(filename);
            List<SaveQWeights<O>> data = mapper.readValue(file, type2);
            Map<O, List<Double>> qValues = data.stream().collect(Collectors.toMap(SaveQWeights::getObservation, SaveQWeights::getWeights));
            for (Map.Entry<O, List<Double>> entry : qValues.entrySet()) {
                MultiFileQNetwork<O, A> network = new MultiFileQNetwork<>(0.1, actionClass);
                network.train(entry.getKey(), entry.getValue().stream().mapToDouble(Double::doubleValue).toArray());
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Getter
    @Setter
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SaveData<O extends Observation> {
        private O observation;
        private List<Double> weights;
    }
}
