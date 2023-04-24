package org.example.entity.torch;

public class Loss {

    public double[][] backward(double[][] output, double[][] target) {
        int numRows = output.length;
        int numCols = output[0].length;
        double[][] gradOutput = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                gradOutput[i][j] = 2.0 * (output[i][j] - target[i][j]) / numRows;
            }
        }

        return gradOutput;
    }

    public double forward(double[][] output, double[][] target) {
        int numRows = output.length;
        int numCols = output[0].length;
        double loss = 0.0;

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double diff = output[i][j] - target[i][j];
                loss += diff * diff;
            }
        }

        return loss / (numRows * numCols);
    }

}
