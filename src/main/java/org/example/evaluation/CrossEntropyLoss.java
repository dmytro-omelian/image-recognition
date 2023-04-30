package org.example.evaluation;

public class CrossEntropyLoss {

    private int numClasses;
    private double[] expScores;
    private double[] probs;
    private double loss;
    private double[][] gradInputs;

    public CrossEntropyLoss(int numClasses) {
        this.numClasses = numClasses;
    }

    public double forward(double[][] inputs, int[] labels) {
        int m = inputs.length;
        expScores = new double[m];
        probs = new double[m * numClasses];
        loss = 0.0;
        for (int i = 0; i < m; i++) {
            double maxScore = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < numClasses; j++) {
                double score = inputs[i][j];
                if (score > maxScore) {
                    maxScore = score;
                }
            }
            double sumExpScores = 0.0;
            for (int j = 0; j < numClasses; j++) {
                expScores[i] = Math.exp(inputs[i][j] - maxScore);
                sumExpScores += expScores[i];
            }
            for (int j = 0; j < numClasses; j++) {
                int index = i * numClasses + j;
                probs[index] = expScores[i] / sumExpScores;
                if (j == labels[i]) {
                    loss += -Math.log(probs[index]);
                }
            }
        }
        loss /= m;
        return loss;
    }

    public double[][] backward(double[][] inputs, int[] labels) {
        int m = inputs.length;
        gradInputs = new double[m][numClasses];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < numClasses; j++) {
                int index = i * numClasses + j;
                gradInputs[i][j] = probs[index];
                if (j == labels[i]) {
                    gradInputs[i][j] -= 1;
                }
                gradInputs[i][j] /= m;
            }
        }
        return gradInputs;
    }

}
