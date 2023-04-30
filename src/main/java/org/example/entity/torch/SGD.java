package org.example.entity.torch;

import java.util.Arrays;

public class SGD {
    private Linear model;
    private double learningRate;

    public SGD(Linear model, double learningRate) {
        this.model = model;
        this.learningRate = learningRate;
    }

    public void zero_grad() {
        for (int i = 0; i < model.gradWeight.length; i++) {
            for (int j = 0; j < model.gradWeight[0].length; j++) {
                model.gradWeight[i][j] = 0.0;
            }
        }

        Arrays.fill(model.gradBias, 0.0);
    }

    public void step() {
        for (int i = 0; i < model.weight.length; i++) {
            for (int j = 0; j < model.weight[0].length; j++) {
                model.weight[i][j] -= learningRate * model.gradWeight[i][j];
            }
        }

        for (int i = 0; i < model.bias.length; i++) {
            model.bias[i] -= learningRate * model.gradBias[i];
        }
    }
}
