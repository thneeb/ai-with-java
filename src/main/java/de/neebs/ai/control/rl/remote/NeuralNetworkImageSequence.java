package de.neebs.ai.control.rl.remote;

import de.neebs.ai.control.rl.Action;
import de.neebs.ai.control.rl.ObservationImageSequence;
import de.neebs.ai.control.rl.QNetwork;
import de.neebs.ai.control.rl.TrainingData;
import de.neebs.aiwithjava.nn.client.entity.TrainingData3D;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class NeuralNetworkImageSequence<A extends Action, O extends ObservationImageSequence> extends AbstractRemoteNetwork<A, O> {
    public NeuralNetworkImageSequence(RemoteNetworkFacade remoteNetworkFacade) {
        super(remoteNetworkFacade);
    }

    public NeuralNetworkImageSequence(RemoteNetworkFacade remoteNetworkFacade, String filename) {
        super(remoteNetworkFacade, filename);
    }

    @Override
    public double[] predict(O observation) {
        List<Double> output = getRemoteNetworkFacade().predict(getInstanceId(), stackImagesSingleChannel(observation.getObservation()));
        return output.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private List<List<List<Double>>> stackImagesSingleChannel(List<BufferedImage> images) {
        if (images == null || images.isEmpty()) {
            return new ArrayList<>();
        }

        // Alle Bilder haben dieselbe Breite und Höhe
        int width = images.get(0).getWidth();
        int height = images.get(0).getHeight();
        int nImages = images.size();

        // Für jedes Bild holen wir alle Pixel in einem Blockaufruf
        List<int[]> pixelArrays = new ArrayList<>(nImages);
        for (BufferedImage img : images) {
            int[] pixels = new int[width * height];
            img.getRGB(0, 0, width, height, pixels, 0, width);
            pixelArrays.add(pixels);
        }

        // Erstelle die verschachtelte Liste in der Reihenfolge:
        // Channel (nImages) x Height x Width
        List<List<List<Double>>> stackedImage = new ArrayList<>(nImages);
        // Für jedes Bild (also jeden Kanal) ...
        for (int i = 0; i < nImages; i++) {
            List<List<Double>> channelData = new ArrayList<>(height);
            int[] pixels = pixelArrays.get(i);
            // Für jede Zeile (Height)
            for (int y = 0; y < height; y++) {
                List<Double> row = new ArrayList<>(width);
                // Für jede Spalte (Width)
                for (int x = 0; x < width; x++) {
                    int index = y * width + x;
                    int rgb = pixels[index];
                    // Hier wird der Rotkanal extrahiert (als Beispiel)
                    int red = (rgb >> 16) & 0xFF;
                    double normalizedRed = red / 255.0;
                    row.add(normalizedRed);
                }
                channelData.add(row);
            }
            stackedImage.add(channelData);
        }
        return stackedImage;
    }
    @Override
    public void train(O observation, double[] target) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void train(List<TrainingData<O>> trainingData) {
        List<TrainingData3D> request = new ArrayList<>();
        for (TrainingData<O> data : trainingData) {
            TrainingData3D trainingData3D = new TrainingData3D();
            trainingData3D.setObservation(stackImagesSingleChannel(data.getObservation().getObservation()));
            trainingData3D.setOutput(Arrays.stream(data.getOutput()).boxed().collect(Collectors.toList()));
            trainingData3D.setAction(data.getAction());
            request.add(trainingData3D);
        }
        getRemoteNetworkFacade().train(getInstanceId(), request);
    }

    @Override
    public QNetwork<A, O> copy() {
        QNetwork<A, O> network = new NeuralNetworkImageSequence<>(getRemoteNetworkFacade());
        network.copyParams(this);
        return network;
    }
}
