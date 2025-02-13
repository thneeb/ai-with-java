package de.neebs.ai.control.rl.remote;

import de.neebs.ai.control.rl.Observation;
import de.neebs.ai.control.rl.QNetwork;
import de.neebs.aiwithjava.nn.client.entity.InstanceConfiguration;
import de.neebs.aiwithjava.nn.client.entity.InstanceId;
import lombok.Getter;

@Getter
public abstract class AbstractRemoteNetwork<O extends Observation> implements QNetwork<O> {
    private final RemoteNetworkFacade remoteNetworkFacade;
    private final String instanceId;

    AbstractRemoteNetwork(RemoteNetworkFacade remoteNetworkFacade) {
        this.remoteNetworkFacade = remoteNetworkFacade;
        this.instanceId = remoteNetworkFacade.createInstance(InstanceConfiguration.builder().initialize(true).build()).getInstanceId();
    }

    @Override
    public void save(String filename) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copyParams(QNetwork<O> source) {
        getRemoteNetworkFacade().copyParams(getInstanceId(), new InstanceId(((AbstractRemoteNetwork<O>) source).getInstanceId()));
    }
}
