package org.parallel_mnist.entity;

import org.parallel_mnist.service.LossService;

public class BatchTask extends Thread {

    private final double[][] trainImages;
    private final int[] trainTarget;
    private final Gradients gradients;
    private final LossService lossService;
    private final Weights weights;

    private final int numFeatures;

    public BatchTask(DataLoader.Batch batch, Gradients gradients, LossService lossService, Weights weights, int numFeatures) {
        this.trainImages = batch.images();
        this.trainTarget = batch.labels();
        this.lossService = lossService;
        this.weights = weights;
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
