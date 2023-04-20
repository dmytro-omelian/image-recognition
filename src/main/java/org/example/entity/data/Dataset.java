package org.example.entity.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Dataset {

    private final String path;
    private List<List<Integer>> features;
    private List<Integer> labels;

    public Dataset(String path) {
        this.path = path;
        this.labels = new ArrayList<>();
    }

    public List<List<Integer>> load() {
        String line = "";
        var train = new ArrayList<List<Integer>>();
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
                var pixels = new ArrayList<Integer>();
                for (int i = 1; i < strPixels.length; ++ i) {
                    var pixel = Integer.parseInt(strPixels[i]);
                    pixels.add(pixel);
                }
                train.add(pixels);
            }
            this.features = train;
            return train;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e); // FIXME handle exceptions in a better way
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public List<List<Integer>> headTrainX() {
        return headTrainX(5);
    }

    public List<List<Integer>> headTrainX(int len) {
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < len; ++ i) {
            var item = this.features.get(i);
            result.add(item);
        }
        return result;
    }

    public List<Integer> headTrainY() {
        return headTrainY(5);
    }

    public List<Integer> headTrainY(int len) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < len; ++ i) {
            var item = this.labels.get(i);
            result.add(item);
        }
        return result;
    }

    public List<List<Integer>> getFeatures() {
        return features;
    }

    public List<Integer> getLabels() {
        return labels;
    }

    public List<List<Double>> getNormalizedFeatures() { // FIXME rewrite with streams
        int n = this.features.size();
        List<List<Double>> result = new ArrayList<>();
        for (List<Integer> feature : this.features) {
            List<Double> temp = new ArrayList<>();
            for (var item : feature) {
                double value = 1.0 * item / 255;
                temp.add(value);
            }
            result.add(temp);
        }
        return result;
    }
}
