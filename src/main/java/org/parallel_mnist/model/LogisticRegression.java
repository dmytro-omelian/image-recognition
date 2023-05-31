package org.parallel_mnist.model;


import org.parallel_mnist.entity.BatchTask;
import org.parallel_mnist.entity.DataLoader;
import org.parallel_mnist.entity.Gradients;
import org.parallel_mnist.entity.Weights;
import org.parallel_mnist.service.LossService;

import java.util.List;
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
        var trainLoader = new DataLoader(X, y, 4096);

        int numInstances = X.size();
        int numFeatures = X.get(0).length;

        for (int iteration = 0; iteration < numIterations; iteration++) {
            Gradients gradients = new Gradients(numFeatures);
            var batches = trainLoader.getBatches();

            ExecutorService executorService = Executors.newFixedThreadPool(15);
            for (DataLoader.Batch batch : batches) {
                Runnable batchTask = new BatchTask(batch, gradients, lossService, weights, numFeatures);
                executorService.execute(batchTask);
            }

            executorService.shutdown();
            try {
                executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                System.err.println("Error occurred while waiting for executor service, see: " + e);
            }

            weights.update(gradients.getGradients(), learningRate, numInstances);
        }
    }
}