package de.neebs.ai.control.rl.remote;

import de.neebs.aiwithjava.nn.client.boundary.DefaultApi;
import de.neebs.aiwithjava.nn.client.entity.*;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.text.MessageFormat;
import java.util.List;

@Service
@RequiredArgsConstructor
public class RemoteNetworkFacade implements DefaultApi {
    private final RestTemplate restTemplate;

    @Value("${app.ai.python.url}")
    private String url;

    @Override
    public InstanceId createInstance(InstanceConfiguration config) {
        String url = this.url + "/instances";
        return restTemplate.postForObject(url, config, InstanceId.class);
    }

    @Override
    public InstanceId getInstance(String instanceId) {
        String url = MessageFormat.format(this.url + "/instances/{0}", instanceId);
        ResponseEntity<InstanceId> response = restTemplate.exchange(url, HttpMethod.GET, null, new ParameterizedTypeReference<>() {});
        if (response.getBody() == null) {
            throw new IllegalStateException("No response");
        }
        return response.getBody();
    }

    @Override
    public List<Double> predict(String instanceId, List<List<List<Double>>> observation3D) {
        String url = MessageFormat.format(this.url + "/instances/{0}/predictions", instanceId);
        ResponseEntity<List<Double>> response = restTemplate.exchange(url, HttpMethod.POST, new HttpEntity<>(observation3D), new ParameterizedTypeReference<>() {});
        if (response.getBody() == null) {
            throw new IllegalStateException("No response");
        }
        return response.getBody();
    }

    @Override
    public void train(String instanceId, List<TrainingData3D> trainingData3D) {
        String url = MessageFormat.format(this.url + "/instances/{0}/trainings", instanceId);
        restTemplate.postForObject(url, new HttpEntity<>(trainingData3D), Void.class);
    }

    @Override
    public void copyParams(String targetInstanceId, InstanceId sourceInstanceId) {
        String url = MessageFormat.format(this.url + "/instances/{0}/copies", targetInstanceId);
        restTemplate.put(url, sourceInstanceId);

    }

    @Override
    public InstanceId copy(String instanceId) {
        String url = MessageFormat.format(this.url + "/instances/{0}/copies", instanceId);
        return restTemplate.postForObject(url, null, InstanceId.class);
    }
}
