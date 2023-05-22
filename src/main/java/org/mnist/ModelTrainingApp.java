package org.mnist;

import org.mnist.entity.data.Dataset;
import org.mnist.model.LogisticRegression;
import org.mnist.preprocessing.PreprocessService;
import org.mnist.service.FileManagerService;
import org.mnist.service.TypeConverterService;

import java.util.List;

public class ModelTrainingApp {
    private static final int NUM_FEATURES = 784; // Number of features in each image
    private static final double LEARNING_RATE = 0.01;
    private static final int NUM_EPOCHS = 10;
    private static final int NUM_ITERATIONS = 100;
    private static final int PRINT_INTERVAL = 2;

    public static void main(String[] args) {
        // Load the MNIST dataset
        Dataset train = new Dataset("src/main/java/org/mnist/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();

        // FIXME features -> it is double values (especially after normalization)

        var features = train.getNormalizedFeatures();
        var labels = train.getLabels();

        PreprocessService.TrainTestEntity trainTestEntity = PreprocessService.trainTestSplit(
                features, labels, 0.2);

        var X_train = trainTestEntity.getTrainX();
        var y_train = trainTestEntity.getTrainY();
        var X_test = trainTestEntity.getTestX();
        var y_test = trainTestEntity.getTestY();

        // Convert the dataset to double arrays
        List<double[]> X_train_double = TypeConverterService.convertToListOfDoubleArrays(X_train);
        List<double[]> X_test_double = TypeConverterService.convertToListOfDoubleArrays(X_test);

        // Create and train the logistic regression model
        LogisticRegression logisticRegression = new LogisticRegression(NUM_FEATURES, LEARNING_RATE, NUM_ITERATIONS);
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            logisticRegression.train(X_train_double, y_train);

            if ((epoch + 1) % PRINT_INTERVAL == 0) {
                double loss = logisticRegression.calculateLoss(X_train_double, y_train);
                System.out.println("Iteration: " + (epoch + 1) + ", Loss: " + loss);
            }
        }

        // Evaluate the model on the test dataset
        int numCorrect = 0;
        int numInstances = X_test_double.size();
        for (int i = 0; i < numInstances; i++) {
            double[] instance = X_test_double.get(i);
            int label = y_test.get(i);

            int prediction = logisticRegression.predict(instance);
            if (prediction == label) {
                numCorrect++;
            }
        }

        double accuracy = (double) numCorrect / numInstances;
        System.out.println("Test Accuracy: " + accuracy);

        FileManagerService fileManagerService = new FileManagerService();
        fileManagerService.saveWeights(logisticRegression.getWeights());
        System.out.println("Weights were saved successfully!");
    }

}