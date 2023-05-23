package org.mnist;

import org.mnist.entity.Dataset;
import org.mnist.service.*;

public class TestModelApp {

    public static void main(String[] args) {
        FileManagerService fileManagerService = new FileManagerService();
        double[][] weights = fileManagerService.readWeights();
        if (weights == null) {
            throw new RuntimeException("Oooops...");
        }

        Dataset train = new Dataset("src/main/java/org/mnist/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();

        var features = train.getFeatures();
        var X = TypeConverterService.convertToArrayOfDoubleArrays(features);

        var testDigit = X[7];
        ImageVisualizationService visualization = new ImageVisualizationService();
        visualization.visualize(testDigit, 28);

        ActivationFunctionService activationFunction = new ActivationFunctionService();
        LossService lossService = new LossService(activationFunction);
        PredictionService predictionService = new PredictionService(lossService);
        int predictedValue = predictionService.predict(testDigit, weights);
        System.out.println("Our prediction is " + predictedValue);
    }

}
