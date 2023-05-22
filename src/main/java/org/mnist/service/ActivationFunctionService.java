package org.mnist.service;

import java.util.Arrays;
import java.util.OptionalDouble;

public class ActivationFunctionService {

    public ActivationFunctionService() {

    }

    public double[] softmax(double[] scores) {
        OptionalDouble value;
        double maxScore = (value = Arrays.stream(scores).max()).isPresent() ? value.getAsDouble() : 0.0;

        double sum = Arrays.stream(scores)
                .map(score -> Math.exp(score - maxScore))
                .sum();

        return Arrays.stream(scores)
                .map(score -> Math.exp(score - maxScore) / sum)
                .toArray();
    }

}
