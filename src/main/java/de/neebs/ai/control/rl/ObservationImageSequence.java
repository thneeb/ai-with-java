package de.neebs.ai.control.rl;

import java.awt.image.BufferedImage;
import java.util.List;

public interface ObservationImageSequence extends Observation {
    List<BufferedImage> getObservation();
}
