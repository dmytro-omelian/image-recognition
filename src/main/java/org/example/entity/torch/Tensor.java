package org.example.entity.torch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Tensor {

    private final double[][] values;
    private final Random random;

    public Tensor(List<List<Double>> X) {
        this.values = new double[X.size()][X.get(0).size()];
        for (int i = 0; i < X.size(); ++ i) {
            for (int j = 0; j < X.get(0).size(); ++ j) {
                this.values[i][j] = X.get(i).get(j);
            }
        }
        this.random = new Random();
    }

    public Tensor(int height, int width, boolean randomInit) {
        if (randomInit) {
            this.values = randInit(height, width);
        } else {
            this.values = initZeros(height, width);
        }

        this.random = new Random();
    }

    public Tensor(double[][] values) {
        this.values = values;
        this.random = new Random();
    }

    private double[][] initZeros(int height, int width) {
        var result = new double[height][width];
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++ j) {
                result[i][j] = 0.0;
            }
        }
        return result;
    }

    private double[][] randInit(int height, int width) {
        var result = new double[height][width];
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                result[i][j] = random.nextDouble();
            }
        }
        return result;
    }

    public Shape getShape() {
        int height = this.values.length;
        int width = (this.values.length > 0) ? this.values[0].length : 0;
        return new Shape(height, width);
    }

    /***
     *
     * @return tensor
     */
    public Tensor T() {
        var shape = this.getShape();
        Tensor transposed = new Tensor(shape.getWidth(), shape.getHeight(), false);
        for (int i = 0; i < shape.getWidth(); ++ i) {
            for (int j = 0; j < shape.getHeight(); ++ j) {
                transposed.set(i, j, this.values[i][j]);
            }
        }
        return transposed;
    }

    private void set(int i, int j, Double value) {
        this.values[i][j] = value;
    }

    /***
     *
     * @param bias
     *  dimension: n x 1
     *
     * current tensor size:
     */
    public void add(Tensor bias) {
        Shape shape = bias.getShape();

        int batchSize = shape.getHeight();
        int outputDim = shape.getWidth();
        var biasValues = bias.getTensorValues();
        for (int i = 0; i < batchSize; ++ i) {
            for (int j = 0; j < outputDim; ++ j) {
                double value = bias.getTensorValues()[i][j];
                this.values[i][j] += value;
            }
        }
    }

    private double[][] getTensorValues() {
        return this.values;
    }

    public Tensor multiply(Tensor b) {
        var input = this.values;
        var weight = b.getTensorValues();

        int batchSize = input.length;
        int inFeatures = input[0].length;
        int outFeatures = weight.length;

        double[][] output = new double[batchSize][outFeatures];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outFeatures; j++) {
                output[i][j] = 0;
                for (int k = 0; k < inFeatures; k++) {
                    output[i][j] += input[i][k] * weight[j][k];
                }
            }
        }

        return new Tensor(output);
    }
}
