package org.parallel_mnist.model;


import org.parallel_mnist.entity.DataLoader;
import org.parallel_mnist.entity.Weights;
import org.parallel_mnist.service.LossService;

import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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

//        int numThreads = Thread.activeCount();
//        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        Thread[] tasks = new Thread[numIterations];

        for (int iteration = 0; iteration < numIterations; iteration++) {
            IterationTask iterationTask = new IterationTask(trainLoader.getBatches(), numFeatures, numInstances);
            tasks[iteration] = iterationTask;
        }

        for (int iteration = 0; iteration < numIterations; iteration++) {
            tasks[iteration].start();
        }

        for (int iteration = 0; iteration < numIterations; iteration++) {
            try {
                tasks[iteration].join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

    }

    public class IterationTask extends Thread {

        private final List<DataLoader.Batch> batches;
        private final int numFeatures;
        private final int numInstances;

        public IterationTask(List<DataLoader.Batch> batches, int numFeatures, int numInstances) {
            this.batches = batches;
            this.numFeatures = numFeatures;
            this.numInstances = numInstances;
        }

        @Override
        public void run() {
            double[][] gradients = new double[10][numFeatures];

            for (DataLoader.Batch value : batches) {
                var trainImages = value.images();
                var trainTarget = value.labels();

                for (int i = 0; i < trainImages.length; i++) {
                    double[] instance = trainImages[i];
                    int label = trainTarget[i];

                    double[] probabilities = lossService.calculateProbs(weights, numFeatures, instance);
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
            }
            weights.update(gradients, learningRate, numInstances);
        }

    }
}