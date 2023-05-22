package org.mnist;

import org.mnist.entity.Dataset;
import org.mnist.service.FileManagerService;
import org.mnist.service.ImageVisualizationService;
import org.mnist.service.TypeConverterService;

import java.util.Arrays;
import java.util.OptionalDouble;

public class TestModelApp {

    public static void main(String[] args) {
        FileManagerService fileManagerService = new FileManagerService();
        double[][] weights = fileManagerService.readWeights();
        if (weights == null) {
            throw new RuntimeException("Oooops...");
        }

        Dataset train = new Dataset("src/main/java/org/mnist/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();

        var features = train.getFeatures();
        var X = TypeConverterService.convertToArrayOfDoubleArrays(features);

        var testDigit = X[7];
        ImageVisualizationService visualization = new ImageVisualizationService();
        visualization.visualize(testDigit, 28);

        int predictedValue = predict(testDigit, weights);
        System.out.println("Our prediction is " + predictedValue);
    }

    public static int predict(double[] X, double[][] weights) {
        int numFeatures = X.length;
        double[] scores = new double[10];

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < numFeatures; j++) {
                scores[i] += weights[i][j] * X[j];
            }
        }

        double[] probabilities = softmax(scores);

        int maxIndex = 0;
        double maxProbability = probabilities[0];

        for (int i = 1; i < 10; i++) {
            if (probabilities[i] > maxProbability) {
                maxIndex = i;
                maxProbability = probabilities[i];
            }
        }

        return maxIndex;
    }

    private static double[] softmax(double[] scores) {
        OptionalDouble value;
        double maxScore = (value = Arrays.stream(scores).max()).isPresent() ? value.getAsDouble() : 0.0;

        double sum = Arrays.stream(scores)
                .map(score -> Math.exp(score - maxScore))
                .sum();

        return Arrays.stream(scores)
                .map(score -> Math.exp(score - maxScore) / sum)
                .toArray();
    }

}
