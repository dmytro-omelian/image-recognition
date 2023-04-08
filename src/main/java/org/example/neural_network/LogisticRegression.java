package org.example.neural_network;

import org.example.java_torch.Linear;

public class LogisticRegression implements Module {

    private final int inputDim; // 28 * 28 - size of images
    private final int outputDim; // number of labels. In our case it's 10 (from 0 to 9)
    private final Linear linear;

    public LogisticRegression(int inputDim, int outputDim) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.linear = new Linear(inputDim, outputDim);
    }

    @Override
    public void forward(/*int[][] x*/) {
        // linear.calculate(/*   x    */);
        // return out
    }

}
