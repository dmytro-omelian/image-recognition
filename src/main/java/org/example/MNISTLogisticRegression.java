package org.example;

import org.example.entity.data.Dataset;
import org.example.preprocessing.PreprocessService;

import java.util.List;
import java.util.stream.Collectors;

public class MNISTLogisticRegression {
    private static final int NUM_FEATURES = 784; // Number of features in each image
    private static final double LEARNING_RATE = 0.01;
    private static final int NUM_EPOCHS = 10;
    private static final int NUM_ITERATIONS = 100;
    private static final int PRINT_INTERVAL = 3;

    public static List<double[]> convertToDoubleArray(List<List<Double>> list) {
        return list.stream()
                .map(innerList -> innerList.stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray())
                .collect(Collectors.toList());
    }

    public static void main(String[] args) {
        // Load the MNIST dataset
        Dataset train = new Dataset("src/main/java/org/example/data/train.csv"); // FIXME add parameter load on start dataset
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
        for (int iteration = 0; iteration < NUM_EPOCHS; iteration++) {
            logisticRegression.train(X_train_double, y_train);

            if ((iteration + 1) % PRINT_INTERVAL == 0) {
                double loss = logisticRegression.calculateLoss(X_train_double, y_train);
                System.out.println("Iteration: " + (iteration + 1) + ", Loss: " + loss);
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
    }
}
