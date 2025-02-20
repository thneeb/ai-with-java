package de.neebs.ai.control.rl.remote;

import de.neebs.ai.control.rl.Action;
import de.neebs.ai.control.rl.Observation;
import de.neebs.ai.control.rl.QNetwork;
import de.neebs.ai.control.rl.Transition;
import de.neebs.aiwithjava.nn.client.entity.InstanceConfiguration;
import de.neebs.aiwithjava.nn.client.entity.InstanceId;
import lombok.Getter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Getter
public abstract class AbstractRemoteNetwork<A extends Action, O extends Observation> implements QNetwork<A, O> {
    private final RemoteNetworkFacade remoteNetworkFacade;
    private final String instanceId;

    AbstractRemoteNetwork(RemoteNetworkFacade remoteNetworkFacade) {
        this.remoteNetworkFacade = remoteNetworkFacade;
        this.instanceId = remoteNetworkFacade.createInstance(InstanceConfiguration.builder().build()).getInstanceId();
    }

    AbstractRemoteNetwork(RemoteNetworkFacade remoteNetworkFacade, String filename) {
        this.remoteNetworkFacade = remoteNetworkFacade;
        String instanceId = readInstanceId(filename);
        this.instanceId = remoteNetworkFacade.getInstance(instanceId).getInstanceId();
    }

    private String readInstanceId(String filename) {
        Path path = Paths.get(filename);
        try {
            return Files.readString(path);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void save(String filename) {
        try {
            Files.writeString(Paths.get(filename), instanceId);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void copyParams(QNetwork<A, O> source) {
        getRemoteNetworkFacade().copyParams(getInstanceId(), new InstanceId(((AbstractRemoteNetwork<A, O>) source).getInstanceId()));
    }
}
