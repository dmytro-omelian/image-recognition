package org.parallel_mnist.entity;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Dataset {

    private final String path;
    private final List<Integer> labels;
    private List<List<Double>> features;

    public Dataset(String path) {
        this.path = path;
        this.labels = new ArrayList<>();
    }

    public void load() {
        String line = "";
        var train = new ArrayList<List<Double>>();
        try {
            boolean labelsSkipped = false;
            BufferedReader br = new BufferedReader(new FileReader(this.path));
            while ((line = br.readLine()) != null) {
                if (!labelsSkipped) { // FIXME it is bullshit to have smth like that in code
                    labelsSkipped = true;
                    continue;
                }
                String[] strPixels = line.split(",");
                this.labels.add(Integer.parseInt(strPixels[0]));
                var pixels = new ArrayList<Double>();
                for (int i = 1; i < strPixels.length; ++i) {
                    var pixel = Double.parseDouble(strPixels[i]);
                    pixels.add(pixel);
                }
                train.add(pixels);
            }
            this.features = train;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e); // FIXME handle exceptions in a better way
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public List<List<Double>> getFeatures() {
        return features;
    }

    public List<Integer> getLabels() {
        return labels;
    }

    public List<List<Double>> getNormalizedFeatures() {
        return this.features.parallelStream().map(featureList ->
                        featureList.stream().map(this::normalizeFeature).toList())
                .toList();
    }

    private Double normalizeFeature(double feature) {
        return feature / 255;
    }
}
