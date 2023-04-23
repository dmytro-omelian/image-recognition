package org.example.entity.torch;

public class Linear {

    private final int inputDim;
    private final int outputDim;

    private final Tensor weights;
    private final Tensor bias;

    public Linear(int inputDim, int outputDim) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;

        this.weights = new Tensor(outputDim, inputDim, true);
        this.bias = new Tensor(outputDim, 1, true);
    }

    public Tensor forward(Tensor input) {
        Shape shape = input.getShape();
        if (shape.getWidth() != inputDim) {
            throw new RuntimeException("error...");
        }

        Tensor output = multiply(input, this.weights.T());
        output.add(bias);
        return output;
    }

    public Tensor multiply(Tensor a, Tensor b) {
        return a.multiply(b);
    }
}
