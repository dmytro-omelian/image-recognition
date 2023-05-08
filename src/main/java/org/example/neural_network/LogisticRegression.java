package org.example.neural_network;

import org.example.entity.torch.Linear;
import org.example.entity.torch.Tensor;

public class LogisticRegression {

    private final int inputDim; // 28 * 28 - size of images
    private final int outputDim; // number of labels. In our case it's 10 (from 0 to 9)
    public final Linear linear;

    public LogisticRegression(int inputDim, int outputDim) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.linear = new Linear(inputDim, outputDim);
    }

    public double[][] forward(double[][] input) {
        var output = linear.forward(input);
        return output;
    }

}
