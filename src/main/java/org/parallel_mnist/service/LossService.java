package org.parallel_mnist.service;

import org.parallel_mnist.entity.Weights;

import java.util.List;

public class LossService {

    private final ActivationFunctionService activationFunction;

    public LossService(ActivationFunctionService activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double calculateLoss(List<double[]> X, List<Integer> y, Weights w) {
        int numInstances = X.size();
        int numFeatures = X.get(0).length;
        double loss = 0.0;

        for (int i = 0; i < numInstances; i++) {
            double[] instance = X.get(i);
            int label = y.get(i);

            double[] probabilities = calculateProbs(w, numFeatures, instance);

            // Calculate the cross-entropy loss
            double correctProbability = probabilities[label];
            loss += -Math.log(correctProbability);
        }

        loss /= numInstances;

        return loss;
    }

    public double[] calculateProbs(Weights w, int numFeatures, double[] instance) {
        var weights = w.getWeights();
        double[] scores = new double[10];

        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < numFeatures; k++) {
                scores[j] += weights[j][k] * instance[k];
            }
        }

        return activationFunction.softmax(scores);
    }


}
