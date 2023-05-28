package org.parallel_mnist;

import org.parallel_mnist.entity.Dataset;
import org.parallel_mnist.entity.Weights;
import org.parallel_mnist.service.*;

public class TestModelApp {

    public static void main(String[] args) {
        FileManagerService fileManagerService = new FileManagerService();
        Weights w;
        double[][] weights = fileManagerService.readWeights();
        if (weights == null) {
            throw new RuntimeException("Oooops...");
        } else {
            w = new Weights(weights);
        }

        Dataset train = new Dataset("src/main/java/org/mnist/data/train.csv");
        train.load();

        var features = train.getFeatures();
        var X = TypeConverterService.convertToArrayOfDoubleArrays(features);

        var testDigit = X[10];
        ImageVisualizationService visualization = new ImageVisualizationService();
        visualization.visualize(testDigit, 28);

        ActivationFunctionService activationFunction = new ActivationFunctionService();
        LossService lossService = new LossService(activationFunction);
        PredictionService predictionService = new PredictionService(lossService);
        int predictedValue = predictionService.predict(testDigit, w);
        System.out.println("Our prediction is " + predictedValue);
    }

}
