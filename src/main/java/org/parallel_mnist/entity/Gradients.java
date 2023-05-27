package org.parallel_mnist.entity;

import java.util.concurrent.locks.ReentrantLock;

public class Gradients {

    private final ReentrantLock lock = new ReentrantLock();

    private final int numFeatures;
    private final double[][] gradients;

    public Gradients(int numFeatures) {
        this.gradients = new double[10][numFeatures];
        this.numFeatures = numFeatures;
    }

    public void update(int j, double gradient, double[] instance) {
        lock.lock();
        try {
            for (int k = 0; k < numFeatures; k++) {
                gradients[j][k] += gradient * instance[k];
            }
        } finally {
            lock.unlock();
        }
    }

    public double[][] getGradients() {
        return this.gradients;
    }
}
