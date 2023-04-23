package org.example.entity.data;

import java.util.ArrayList;
import java.util.List;

public class DataLoader {

    private final List<List<Integer>> X;
    private final List<Integer> y;
    private final int batchSize;
    private final boolean shuffle;

    public DataLoader(List<List<Integer>> X, List<Integer> y, int batchSize) {
        this(X, y, batchSize, false);
    }

    public DataLoader(List<List<Integer>> X, List<Integer> y, int batchSize, boolean shuffle) {
        this.X = X;
        this.y = y;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
    }

    public List<Batch> getBatches() {
        var batches = new ArrayList<Batch>();
        for (int i = 0; i < this.X.size(); i += batchSize) {
            int right = Math.min(i + batchSize, this.X.size());
            var images = new ArrayList<List<Integer>>();
            var labels = new ArrayList<Integer>();
            for (int left = i; left < right; ++left) {
                images.add(this.X.get(left));
                labels.add(this.y.get(left));
            }
            var batch = new Batch(images, labels);
            batches.add(batch);
        }
        return batches;
    }

    // FIXME convert to Record
    // research what is that?
    static class Batch {

        private final List<List<Integer>> images;
        private final List<Integer> labels;

        public Batch(List<List<Integer>> images, List<Integer> labels) {
            this.images = images;
            this.labels = labels;
        }

        public List<List<Integer>> getImages() {
            return images;
        }

        public List<Integer> getLabels() {
            return labels;
        }
    }

}
