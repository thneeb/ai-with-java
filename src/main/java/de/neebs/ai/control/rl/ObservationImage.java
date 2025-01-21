package de.neebs.ai.control.rl;

import java.awt.image.BufferedImage;

public interface ObservationImage extends Observation {
    BufferedImage getObservation();
}
