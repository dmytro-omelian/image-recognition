package org.parallel_mnist;

import org.parallel_mnist.entity.Dataset;
import org.parallel_mnist.model.LogisticRegression;
import org.parallel_mnist.service.*;

import java.util.List;

public class ModelTrainingApp {
    private static final int NUM_FEATURES = 784;
    private static final double LEARNING_RATE = 0.01;
    private static final int NUM_EPOCHS = 10;
    private static final int NUM_ITERATIONS = 100;
    private static final int PRINT_INTERVAL = 5;

    public static void main(String[] args) {
        long start = System.currentTimeMillis();

        Dataset train = new Dataset("src/main/java/org/mnist/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();

        var features = train.getNormalizedFeatures();
        var labels = train.getLabels();

        PreprocessService.TrainTestEntity trainTestEntity = PreprocessService.trainTestSplit(
                features, labels, 0.2,true,  42, 15000);

        var X_train = trainTestEntity.getTrainX();
        var y_train = trainTestEntity.getTrainY();
        var X_test = trainTestEntity.getTestX();
        var y_test = trainTestEntity.getTestY();

        List<double[]> X_train_double = TypeConverterService.convertToListOfDoubleArrays(X_train);
        List<double[]> X_test_double = TypeConverterService.convertToListOfDoubleArrays(X_test);

        long dataLoaded = System.currentTimeMillis();
        System.out.println("Data was loaded for " + (dataLoaded - start) + " ms");

        ActivationFunctionService activationFunction = new ActivationFunctionService();
        LossService lossService = new LossService(activationFunction);
        LogisticRegression model = new LogisticRegression(NUM_FEATURES, LEARNING_RATE, NUM_ITERATIONS, lossService);
        PredictionService predictionService = new PredictionService(lossService);

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            model.train(X_train_double, y_train);

            if ((epoch + 1) % PRINT_INTERVAL == 0) {
                double loss = lossService.calculateLoss(X_train_double, y_train, model.getWeightsObject());
                System.out.println("Iteration: " + (epoch + 1) + ", Loss: " + loss);
            }
        }

        long modelTrained = System.currentTimeMillis();
        System.out.println("Model was trained for " + (modelTrained - dataLoaded) + " ms");

        int numCorrect = 0;
        int numInstances = X_test_double.size();
        for (int i = 0; i < numInstances; i++) {
            double[] instance = X_test_double.get(i);
            int label = y_test.get(i);

            int prediction = predictionService.predict(instance, model.getWeightsObject());
            if (prediction == label) {
                numCorrect++;
            }
        }

        double accuracy = (double) numCorrect / numInstances;
        System.out.println("Test Accuracy: " + accuracy);

        long evaluatedScore = System.currentTimeMillis();
        System.out.println("Model was evaluated for " + (evaluatedScore - modelTrained) + " ms");

//        FileManagerService fileManagerService = new FileManagerService();
//        fileManagerService.saveWeights(model.getWeights());
//        System.out.println("Weights were saved successfully!");

        long end = System.currentTimeMillis();
        System.out.println("Algorithm has worked for " + (end - start) + " ms");
    }

}
