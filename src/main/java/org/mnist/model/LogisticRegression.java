package org.mnist.model;


import org.mnist.entity.DataLoader;

import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.Random;

public class LogisticRegression {
    private final double[][] weights;
    private final double learningRate;
    private final int numIterations;

    public LogisticRegression(int numFeatures, double learningRate, int numIterations) {
        this.learningRate = learningRate;
        this.numIterations = numIterations;

        Random random = new Random();
        weights = new double[10][numFeatures];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < numFeatures; j++) {
                weights[i][j] = random.nextDouble();
            }
        }
    }

    public void train(List<double[]> X, List<Integer> y) {
        var trainLoader = new DataLoader(X, y, 8);

        int numInstances = X.size();
        int numFeatures = X.get(0).length;

        for (int iteration = 0; iteration < numIterations; iteration++) {
            double[][] gradients = new double[10][numFeatures];

            // FIXME separate instances with batches
            var batches = trainLoader.getBatches();
            for (int batch = 0; batch < batches.size(); ++batch) {
                var trainImages = batches.get(batch).images();
                var trainTarget = batches.get(batch).labels();

                for (int i = 0; i < trainImages.length; i++) {
                    double[] instance = trainImages[i];
                    int label = trainTarget[i];

                    double[] scores = new double[10];

                    // Calculate scores
                    for (int j = 0; j < 10; j++) {
                        for (int k = 0; k < numFeatures; k++) {
                            scores[j] += weights[j][k] * instance[k];
                        }
                    }

                    // Calculate probabilities using softmax
                    double[] probabilities = softmax(scores);

                    // FIXME add parallel algorithm
                    // Calculate the gradients
                    for (int j = 0; j < 10; j++) {
                        double gradient = probabilities[j];
                        if (j == label) {
                            gradient -= 1.0;
                        }

                        for (int k = 0; k < numFeatures; k++) {
                            gradients[j][k] += gradient * instance[k];
                        }
                    }
                }

                // FIXME can add parallel algorithm
                for (int i = 0; i < 10; i++) {
                    for (int j = 0; j < numFeatures; j++) {
                        weights[i][j] -= learningRate * gradients[i][j] / numInstances;
                    }
                }
            }
        }
    }

    public int predict(double[] X) {
        int numFeatures = X.length;
        double[] scores = new double[10];

        // Calculate scores
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < numFeatures; j++) {
                scores[i] += weights[i][j] * X[j];
            }
        }

        // Calculate probabilities using softmax
        double[] probabilities = softmax(scores);

        int maxIndex = 0;
        double maxProbability = probabilities[0];

        // Find the index with the highest probability
        for (int i = 1; i < 10; i++) {
            if (probabilities[i] > maxProbability) {
                maxIndex = i;
                maxProbability = probabilities[i];
            }
        }

        return maxIndex;
    }

    private double[] softmax(double[] scores) {
        OptionalDouble value;
        double maxScore = (value = Arrays.stream(scores).max()).isPresent() ? value.getAsDouble() : 0.0;

        double sum = Arrays.stream(scores)
                .map(score -> Math.exp(score - maxScore))
                .sum();

        return Arrays.stream(scores)
                .map(score -> Math.exp(score - maxScore) / sum)
                .toArray();
    }

    public double calculateLoss(List<double[]> X, List<Integer> y) {
        int numInstances = X.size();
        int numFeatures = X.get(0).length;
        double loss = 0.0;

        for (int i = 0; i < numInstances; i++) {
            double[] instance = X.get(i);
            int label = y.get(i);

            double[] scores = new double[10];

            // Calculate scores
            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < numFeatures; k++) {
                    scores[j] += weights[j][k] * instance[k];
                }
            }

            // Calculate probabilities using softmax
            double[] probabilities = softmax(scores);

            // Calculate the cross-entropy loss
            double correctProbability = probabilities[label];
            loss += -Math.log(correctProbability);
        }

        loss /= numInstances;

        return loss;
    }


    public double[][] getWeights() {
        return this.weights;
    }
}