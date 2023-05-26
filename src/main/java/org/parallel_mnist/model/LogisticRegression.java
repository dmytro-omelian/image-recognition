package org.parallel_mnist.model;


import org.parallel_mnist.entity.DataLoader;
import org.parallel_mnist.entity.Weights;
import org.parallel_mnist.service.LossService;

import java.util.List;
import java.util.Random;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class LogisticRegression {
    private final Weights weights;
    private final double learningRate;
    private final int numIterations;

    private final LossService lossService;

    public LogisticRegression(int numFeatures, double learningRate, int numIterations, LossService lossService) {
        this.learningRate = learningRate;
        this.numIterations = numIterations;
        this.lossService = lossService;
        this.weights = new Weights(numFeatures);
    }

    public double[][] getWeights() {
        return this.weights.getWeights();
    }

    public Weights getWeightsObject() {
        return this.weights;
    }

    public void train(List<double[]> X, List<Integer> y) {
        var trainLoader = new DataLoader(X, y, 8);

        int numInstances = X.size();
        int numFeatures = X.get(0).length;

        for (int iteration = 0; iteration < numIterations; iteration++) {
            double[][] gradients = new double[10][numFeatures];

            ExecutorService executorServiceBatches = Executors.newFixedThreadPool(5);

            var batches = trainLoader.getBatches();
            for (DataLoader.Batch value : batches) {
                var trainImages = value.images();
                var trainTarget = value.labels();

                for (int i = 0; i < trainImages.length; i++) {
                    double[] instance = trainImages[i];
                    int label = trainTarget[i];

                    double[] probabilities = lossService.calculateProbs(weights, numFeatures, instance);


                    ExecutorService executorServiceFeatures = Executors.newFixedThreadPool(5);
                    try {
                        for (int j = 0; j < 10; j++) {
                            int finalJ = j;
                            executorServiceFeatures.execute(() -> {
                                double gradient = probabilities[finalJ];
                                if (finalJ == label) {
                                    gradient -= 1.0;
                                }

                                for (int k = 0; k < numFeatures; k++) {
                                    gradients[finalJ][k] += gradient * instance[k];
                                }
                            });
                        }
                    } finally {
                        executorServiceFeatures.shutdown();
                        try {
                            executorServiceFeatures.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    }
                }
                weights.update(gradients, learningRate, numInstances);
            }

            executorServiceBatches.shutdown();
        }
    }
}