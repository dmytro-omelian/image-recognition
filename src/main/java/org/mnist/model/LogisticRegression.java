package org.mnist.model;


import org.mnist.entity.DataLoader;
import org.mnist.service.ActivationFunctionService;
import org.mnist.service.LossService;

import java.util.List;
import java.util.Random;

public class LogisticRegression {
    private final double[][] weights;
    private final double learningRate;
    private final int numIterations;

    private final LossService lossService;

    public LogisticRegression(int numFeatures, double learningRate, int numIterations, LossService lossService) {
        this.learningRate = learningRate;
        this.numIterations = numIterations;
        this.lossService = lossService;

        Random random = new Random();
        weights = new double[10][numFeatures];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < numFeatures; j++) {
                weights[i][j] = random.nextDouble();
            }
        }
    }

    public double[][] getWeights() {
        return this.weights;
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

                    double[] probabilities = lossService.calculateProbs(weights, numFeatures, instance);

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
}