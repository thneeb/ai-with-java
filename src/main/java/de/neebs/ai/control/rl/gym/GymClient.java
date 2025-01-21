package de.neebs.ai.control.rl.gym;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.neebs.ai.control.rl.Observation;
import lombok.Builder;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class GymClient {
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    @Getter
    @Setter
    private static class InstanceId {
        @JsonProperty("instance_id")
        private String instanceId;
    }

    @Getter
    @Setter
    @Builder
    private static class EnvironmentId {
        @JsonProperty("env_id")
        private String environmentId;
    }

    @Getter
    @Setter
    @Builder
    private static class Action {
        private int action;
    }

    @Getter
    @Setter
    @Builder
    private static class ObservationSpace {
        private String name;
        private List<Integer> shape;
    }

    public String makeEnv(String envName) {
        String url = "http://localhost:5000/v1/envs/";
        InstanceId response = restTemplate.postForObject(url, EnvironmentId.builder().environmentId(envName).build(), InstanceId.class);
        if (response == null) {
            throw new IllegalArgumentException("Could not create environment");
        }
        return response.getInstanceId();
    }

    public <T> T reset(String instanceId, Class<T> clazz) {
        String url = "http://localhost:5000/v1/envs/" + instanceId + "/reset/";
        return restTemplate.postForObject(url, null, clazz);
    }

    public <T extends Observation> GymStepResult<T> step(String instanceId, int action, Class<T> clazz) {
        String url = "http://localhost:5000/v1/envs/" + instanceId + "/step/";
        ResponseEntity<Map<String, Object>> response = restTemplate.exchange(url, HttpMethod.POST, new HttpEntity<>(Action.builder().action(action).build()), new ParameterizedTypeReference<>() {});
        if (response.getBody() == null) {
            throw new IllegalArgumentException("Could not step environment");
        }
        Map<String, Object> map = Map.of("observation", response.getBody().get("observation"));
        T observation = objectMapper.convertValue(map, clazz);
        return GymStepResult.<T>builder()
                .observation(observation)
                .reward((double) response.getBody().get("reward"))
                .done((boolean) response.getBody().get("done"))
                .truncated((boolean) response.getBody().get("truncated"))
                .info((Map<String, Object>) response.getBody().get("info"))
                .build();
    }

    public List<Integer> getObservationSpace(String instanceId) {
        String url = "http://localhost:5000/v1/envs/" + instanceId + "/observation_space/";
        ObservationSpace observationSpace = restTemplate.getForObject(url, ObservationSpace.class);
        if (observationSpace == null) {
            throw new IllegalArgumentException("Could not get observation space");
        }
        return observationSpace.getShape();
    }
}
