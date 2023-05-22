package org.mnist.entity.torch;

import java.util.Random;

public class Linear {
    protected double[][] weight;
    protected double[] bias;
    protected double[][] gradWeight;
    protected double[] gradBias;

    public Linear(int inFeatures, int outFeatures) {
        Random random = new Random();

        // Initialize weights and biases with random values
        weight = new double[outFeatures][inFeatures];
        for (int i = 0; i < outFeatures; i++) {
            for (int j = 0; j < inFeatures; j++) {
                weight[i][j] = random.nextDouble();
            }
        }

        bias = new double[outFeatures];
        for (int i = 0; i < outFeatures; i++) {
            bias[i] = random.nextDouble();
        }

        // Initialize gradient buffers with zeros
        gradWeight = new double[outFeatures][inFeatures];
        gradBias = new double[outFeatures];
    }

    public double[][] forward(double[][] input) {
        int batchSize = input.length;
        int inFeatures = input[0].length;
        int outFeatures = weight.length;

        double[][] output = new double[batchSize][outFeatures];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outFeatures; j++) {
                output[i][j] = bias[j];
                for (int k = 0; k < inFeatures; k++) {
                    output[i][j] += input[i][k] * weight[j][k];
                }
            }
        }

        return output;
    }

    public double[][] backward(double[][] input, double[][] gradOutput, double learningRate) {
        int batchSize = input.length;
        int inFeatures = input[0].length;
        int outFeatures = weight.length;

        double[][] gradInput = new double[batchSize][inFeatures];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outFeatures; j++) {
                for (int k = 0; k < inFeatures; k++) {
                    gradInput[i][k] += gradOutput[i][j] * weight[j][k];
                    gradWeight[j][k] += gradOutput[i][j] * input[i][k];
                }
                gradBias[j] += gradOutput[i][j];
            }
        }

        // Update weights and biases
        for (int i = 0; i < outFeatures; i++) {
            for (int j = 0; j < inFeatures; j++) {
                weight[i][j] -= learningRate * gradWeight[i][j];
                gradWeight[i][j] = 0.0;
            }
            bias[i] -= learningRate * gradBias[i];
            gradBias[i] = 0.0;
        }

        return gradInput;
    }
}

