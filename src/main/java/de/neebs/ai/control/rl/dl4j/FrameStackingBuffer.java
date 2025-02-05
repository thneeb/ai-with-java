package de.neebs.ai.control.rl.dl4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class FrameStackingBuffer {
    private final int stackSize; // Anzahl der Frames, die gestackt werden sollen (z. B. 4)
    private final int height;    // Höhe eines Frames
    private final int width;     // Breite eines Frames
    private final int channels;  // Kanäle eines Frames (z. B. 1 für Graustufen)
    private INDArray buffer;     // Puffer für die gestackten Frames

    public FrameStackingBuffer(int stackSize, int height, int width, int channels) {
        this.stackSize = stackSize;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.buffer = Nd4j.zeros(stackSize, height, width, channels);
    }

    // Fügt ein neues Frame hinzu und entfernt das älteste Frame
    public void addFrame(INDArray frame) {
        // Verschiebe die Frames im Puffer
        for (int i = 0; i < stackSize - 1; i++) {
            INDArray nextFrame = buffer.get(NDArrayIndex.point(i + 1), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            buffer.put(new INDArrayIndex[]{NDArrayIndex.point(i)}, nextFrame);
        }
        // Füge das neue Frame hinzu
        buffer.put(new INDArrayIndex[]{NDArrayIndex.point(stackSize - 1)}, frame);
    }

    // Gibt die gestackten Frames zurück
    public INDArray getStackedFrames() {
        return buffer.reshape(1, height, width, channels * stackSize);
    }

    // Setzt den Puffer zurück (z. B. am Anfang einer neuen Episode)
    public void reset() {
        buffer = Nd4j.zeros(stackSize, height, width, channels);
    }
}