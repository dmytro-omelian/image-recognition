package org.mnist.service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PreprocessService {

    public PreprocessService() {
    }

    public static TrainTestEntity trainTestSplit(List<List<Double>> X, List<Integer> y, double testSize) {
        return trainTestSplit(X, y, testSize, true, 42);
    }

    public static TrainTestEntity trainTestSplit(List<List<Double>> X, List<Integer> y, double testSize, boolean shuffle, int randomState) {
        if (testSize <= 0 || testSize >= 1) {
            throw new IllegalStateException("Test size is not valid...");
        }
        int n = X.size();
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            indexes.add(i);
        }
        if (shuffle) {
            Collections.shuffle(indexes);
        }

        List<List<Double>> shuffledX = indexes.stream().map(X::get).toList();
        List<Integer> shuffledY = indexes.stream().map(y::get).toList();

        var trainX = new ArrayList<List<Double>>();
        var trainY = new ArrayList<Integer>();
        var testX = new ArrayList<List<Double>>();
        var testY = new ArrayList<Integer>();

        int numberOfTestItems = (int) (n * testSize);
        for (int i = 0; i < numberOfTestItems; ++i) {
            testX.add(shuffledX.get(i));
            testY.add(shuffledY.get(i));
        }

        for (int i = numberOfTestItems; i < n; ++i) {
            trainX.add(shuffledX.get(i));
            trainY.add(shuffledY.get(i));
        }

        return new TrainTestEntity(
                trainX,
                trainY,
                testX,
                testY
        );
    }

    public static class TrainTestEntity {

        private final List<List<Double>> trainX;
        private final List<Integer> trainY;
        private final List<List<Double>> testX;
        private final List<Integer> testY;

        public TrainTestEntity(List<List<Double>> trainX, List<Integer> trainY,
                               List<List<Double>> testX, List<Integer> testY) {
            this.trainX = trainX;
            this.trainY = trainY;
            this.testX = testX;
            this.testY = testY;
        }

        public List<List<Double>> getTrainX() {
            return trainX;
        }

        public List<Integer> getTrainY() {
            return trainY;
        }

        public List<List<Double>> getTestX() {
            return testX;
        }

        public List<Integer> getTestY() {
            return testY;
        }
    }

}