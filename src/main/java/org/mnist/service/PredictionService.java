package org.mnist.service;

public class PredictionService {

    private final ActivationFunctionService activationFunction;

    public PredictionService(ActivationFunctionService activationFunction) {
        this.activationFunction = activationFunction;
    }

    public int predict(double[] X, double[][] weights) {
        int numFeatures = X.length;
        double[] scores = new double[10];

        // Calculate scores
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < numFeatures; j++) {
                scores[i] += weights[i][j] * X[j];
            }
        }

        // Calculate probabilities using softmax
        double[] probabilities = activationFunction.softmax(scores);

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

}
