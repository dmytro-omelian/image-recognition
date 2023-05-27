package org.parallel_mnist.model;


import org.parallel_mnist.entity.DataLoader;
import org.parallel_mnist.entity.Weights;
import org.parallel_mnist.service.LossService;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

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
        var trainLoader = new DataLoader(X, y, 2048);

        int numInstances = X.size();
        int numFeatures = X.get(0).length;

        for (int iteration = 0; iteration < numIterations; iteration++) {

            var batches = trainLoader.getBatches();
//            System.out.println("number of batches: " + batches.size());

            ExecutorService executorService = Executors.newFixedThreadPool(15);

            Gradients gradients = new Gradients(numFeatures);

            for (DataLoader.Batch batch : batches) {
                Runnable batchTask = new BatchTask(batch, gradients, numFeatures);
                executorService.execute(batchTask);
            }

            executorService.shutdown();

            while (!executorService.isTerminated()) {
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }

            weights.update(gradients.getGradients(), learningRate, numInstances);
        }
    }

    public class Gradients {

        private final ReentrantLock lock = new ReentrantLock();

        private final int numFeatures;
        private final double[][] gradients;

        public Gradients(int numFeatures) {
            this.gradients = new double[10][numFeatures];
            this.numFeatures = numFeatures;
        }

        public void update(int j, double gradient, double[] instance) {
            lock.lock();
            try {
                for (int k = 0; k < numFeatures; k++) {
                    gradients[j][k] += gradient * instance[k];
                }
            } finally {
                lock.unlock();
            }
        }

        public double[][] getGradients() {
            return this.gradients;
        }
    }

    public class BatchTask extends Thread {

        private final double[][] trainImages;
        private final int[] trainTarget;
        private final Gradients gradients;

        private final int numFeatures;

        public BatchTask(DataLoader.Batch batch, Gradients gradients, int numFeatures) {
            this.trainImages = batch.images();
            this.trainTarget = batch.labels();
            this.gradients = gradients;
            this.numFeatures = numFeatures;
        }

        @Override
        public void run() {
            for (int i = 0; i < trainImages.length; i++) {
                double[] instance = trainImages[i];
                int label = trainTarget[i];

                double[] probabilities = lossService.calculateProbs(weights, numFeatures, instance);
                for (int j = 0; j < 10; j++) {
                    double gradient = probabilities[j];
                    if (j == label) {
                        gradient -= 1.0;
                    }
                    gradients.update(j, gradient, instance);
                }
            }
        }
    }
}