package org.mnist;

import org.mnist.entity.data.Dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.stream.Collectors;

public class Main {

    private static final String WEIGHTS_FILENAME = "weights.txt";

    public static void main(String[] args) {
        double[][] weights = readWeights();
        if (weights == null) {
            throw new RuntimeException("Oooops...");
        }

        Dataset train = new Dataset("src/main/java/org/mnist/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();

        var features = train.getFeatures();
        var X = convertToDoubleArray(features);

        var testDigit = X[7];
        ImageVisualization visualization = new ImageVisualization();
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


    private static double[][] readWeights() {
        try (BufferedReader reader = new BufferedReader(new FileReader(WEIGHTS_FILENAME))) {
            String line;
            List<List<Double>> weights = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                List<Double> row;
                String[] values = line.trim().split("\\s+");
                row = Arrays.stream(values).map(Double::parseDouble).collect(Collectors.toList());
                weights.add(row);
            }
            return convertToDoubleArray(weights);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static double[][] convertToDoubleArray(List<List<Double>> list) {
        return list.stream()
                .map(row -> row.stream().mapToDouble(Double::doubleValue).toArray())
                .toArray(double[][]::new);
    }

}
