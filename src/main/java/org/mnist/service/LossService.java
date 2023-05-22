package org.mnist.service;

import java.util.List;

public class LossService {

    private final ActivationFunctionService activationFunction;

    public LossService(ActivationFunctionService activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double calculateLoss(List<double[]> X, List<Integer> y, double[][] weights) {
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
            double[] probabilities = activationFunction.softmax(scores);

            // Calculate the cross-entropy loss
            double correctProbability = probabilities[label];
            loss += -Math.log(correctProbability);
        }

        loss /= numInstances;

        return loss;
    }


}
