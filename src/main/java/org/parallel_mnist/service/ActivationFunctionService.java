package org.parallel_mnist.service;

import java.util.Arrays;
import java.util.OptionalDouble;

public class ActivationFunctionService {

    public ActivationFunctionService() {

    }

    public double[] softmax(double[] scores) {
        OptionalDouble value;
        double maxScore = (value = Arrays.stream(scores).parallel()
                .max()).isPresent() ? value.getAsDouble() : 0.0;

        double sum = Arrays.stream(scores).parallel()
                .map(score -> Math.exp(score - maxScore))
                .sum();

        return Arrays.stream(scores).parallel()
                .map(score -> Math.exp(score - maxScore) / sum)
                .toArray();
    }

}
