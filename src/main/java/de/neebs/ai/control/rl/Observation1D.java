package de.neebs.ai.control.rl;

public interface Observation1D extends Observation{
    double[] getFlattenedObservation();
}
