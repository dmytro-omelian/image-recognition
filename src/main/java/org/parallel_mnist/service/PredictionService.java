package org.parallel_mnist.service;

public class PredictionService {

    private final LossService lossService;

    public PredictionService(LossService lossService) {
        this.lossService = lossService;
    }

    public int predict(double[] X, double[][] weights) {
        int numFeatures = X.length;
        double[] probabilities = lossService.calculateProbs(weights, numFeatures, X);

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

}
