package org.example.entity.data;

import java.util.ArrayList;
import java.util.List;

public class DataLoader {

    private final List<List<Double>> X;
    private final List<Integer> y;
    private final int batchSize;
    private final boolean shuffle;

    public DataLoader(List<List<Double>> X, List<Integer> y, int batchSize) {
        this(X, y, batchSize, false);
    }

    public DataLoader(List<List<Double>> X, List<Integer> y, int batchSize, boolean shuffle) {
        this.X = X;
        this.y = y;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
    }

    public List<Batch> getBatches() {
        var batches = new ArrayList<Batch>();
        for (int i = 0; i < this.X.size(); i += batchSize) {
            int right = Math.min(i + batchSize, this.X.size());
            var images = new ArrayList<List<Double>>();
            var labels = new ArrayList<Integer>();
            for (int left = i; left < right; ++left) {
                images.add(this.X.get(left));
                labels.add(this.y.get(left));
            }

            var batch = new Batch(
                    convertToDoubleArray(images),
                    convertToIntegerArray(labels)
            );
            batches.add(batch);
        }
        return batches;
    }

    private int[] convertToIntegerArray(ArrayList<Integer> labels) {
        int n = labels.size();
        int[] result = new int[n];
        for (int i = 0; i < n; ++ i) {
            result[i] = labels.get(i);
        }
        return result;
    }

    public record Batch(double[][] images, int[] labels) {

    }

    private static double[][] convertToDoubleArray(List<List<Double>> train_x) {
        return train_x.stream()
                .map(l -> l.stream().mapToDouble(Double::doubleValue).toArray())
                .toArray(double[][]::new);
    }
}
