package org.mnist;

import org.mnist.entity.data.Dataset;
import org.mnist.preprocessing.PreprocessService;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class MNISTLogisticRegression {
    private static final int NUM_FEATURES = 784; // Number of features in each image
    private static final double LEARNING_RATE = 0.01;
    private static final int NUM_EPOCHS = 10;
    private static final int NUM_ITERATIONS = 100;
    private static final int PRINT_INTERVAL = 1;

    private static final String WEIGHTS_FILENAME = "weights.txt";

    public static List<double[]> convertToDoubleArray(List<List<Double>> list) {
        return list.stream()
                .map(innerList -> innerList.stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray())
                .collect(Collectors.toList());
    }

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
        List<double[]> X_train_double = convertToDoubleArray(X_train);
        List<double[]> X_test_double = convertToDoubleArray(X_test);

        // Create and train the logistic regression model
        LogisticRegressionMNIST logisticRegression = new LogisticRegressionMNIST(NUM_FEATURES, LEARNING_RATE, NUM_ITERATIONS);
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

        saveWeights(logisticRegression.getWeights());
        System.out.println("Weights were saved successfully!");
    }

    private static void saveWeights(double[][] weights) {

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(WEIGHTS_FILENAME))) {
            for (double[] row : weights) {
                for (double value : row) {
                    writer.write(String.valueOf(value));
                    writer.write(" ");
                }
                writer.newLine();
            }
            System.out.println("Array data saved to " + WEIGHTS_FILENAME);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
