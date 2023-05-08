package org.example.entity.torch;

import org.example.neural_network.LogisticRegression;

import java.util.Arrays;

public class SGD {
    private LogisticRegression model;
    private double learningRate;

    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    public void zero_grad() {
        for (int i = 0; i < model.linear.gradWeight.length; i++) {
            for (int j = 0; j < model.linear.gradWeight[0].length; j++) {
                model.linear.gradWeight[i][j] = 0.0;
            }
        }

        Arrays.fill(model.linear.gradBias, 0.0);
    }

    public void step() {
        for (int i = 0; i < model.linear.weight.length; i++) {
            for (int j = 0; j < model.linear.weight[0].length; j++) {
                model.linear.weight[i][j] -= learningRate * model.linear.gradWeight[i][j];
            }
        }

        for (int i = 0; i < model.linear.bias.length; i++) {
            model.linear.bias[i] -= learningRate * model.linear.gradBias[i];
        }
    }
}
