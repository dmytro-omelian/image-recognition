package org.parallel_mnist.entity;

import java.util.Random;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Weights {
    private final Lock lock = new ReentrantLock();
    private final double[][] weights;

    public Weights(int numFeatures) {
        Random random = new Random();

        weights = new double[10][numFeatures];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < numFeatures; j++) {
                weights[i][j] = random.nextDouble();
            }
        }
    }

    public Weights(double[][] weights) {
        this.weights = weights;
    }

    public void update(double[][] gradients, double learningRate, int numInstances) {
        lock.lock();
        try {
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < gradients[i].length; j++) {
                    weights[i][j] -= learningRate * gradients[i][j] / numInstances;
                }
            }
        } finally {
            lock.unlock();
        }
    }

    public double[][] getWeights() {
        return this.weights;
    }
}
