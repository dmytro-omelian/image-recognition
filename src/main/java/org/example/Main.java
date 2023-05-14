package org.example;

import org.example.entity.data.DataLoader;
import org.example.entity.data.Dataset;
import org.example.entity.torch.Linear;
import org.example.evaluation.CrossEntropyLoss;
import org.example.neural_network.LogisticRegression;
import org.example.preprocessing.PreprocessService;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;


public class Main {
    public static List<double[]> convertToDoubleArray(List<List<Double>> list) {
        return list.stream()
                .map(innerList -> innerList.stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray())
                .collect(Collectors.toList());
    }

    public static void main(String[] args) {
        Dataset train = new Dataset("src/main/java/org/example/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();

        // FIXME features -> it is double values (especially after normalization)

        var features = train.getNormalizedFeatures();
        var labels = train.getLabels();

        PreprocessService.TrainTestEntity trainTestEntity = PreprocessService.trainTestSplit(
                features, labels, 0.2);

        var train_X = trainTestEntity.getTrainX();
        var train_y = trainTestEntity.getTrainY();
        var test_X = trainTestEntity.getTestX();
        var test_y = trainTestEntity.getTestY();

        var trainLoader = new DataLoader(train_X, train_y, 8);
        var testLoader = new DataLoader(test_X, test_y, 8);

        int inputDim = 28 * 28;
        int outputDim = 10;
//        LogisticRegression model = new LogisticRegression(inputDim, outputDim);

        CrossEntropyLoss error = new CrossEntropyLoss(10);

        double learningRate = 0.01;
//        SGD optimizer = new SGD(learningRate);

        var X_train = convertToDoubleArray(train_X);
        var X_test = convertToDoubleArray(test_X);

        LogisticRegression model = new LogisticRegression(inputDim, learningRate, 10);
        model.train(X_train, train_y);

        double result = model.calculateLoss(X_test, test_y);
        System.out.printf("Loss: {%f}", result);

//        Linear linear = new Linear(inputDim, outputDim);
//        int count = 0;
//        int numEpochs = 10;
//        for (int epoch = 1; epoch <= numEpochs; epoch++) {
//            var batches = trainLoader.getBatches();
//            for (int i = 0; i < batches.size(); ++ i) {
//                var trainImages = batches.get(i).images();
//                var trainTarget = batches.get(i).labels();
//
//                var outputs = linear.forward(trainImages);
//
//                var loss = error.forward(outputs, trainTarget);
//
//                linear.backward(trainImages, outputs, learningRate);
//
//                count += 1;
//
//                if (count % 50 == 0) {
//                    var correct = 0;
//                    var total = 0;
//                    var testBatches = testLoader.getBatches();
//                    for (int j = 0; j < testBatches.size(); ++ j) {
//                        var testImages = testBatches.get(j).images();
//                        var testLabels = testBatches.get(j).labels();
//
//                        var testOutputs = linear.forward(testImages);
//
//                        var predicted = getPredictions(testOutputs);
//
//                        total += testLabels.length;
//
//                        for (int k = 0; k < testLabels.length; ++ k) {
//                            if (Objects.equals(testLabels[k], predicted[k])) {
//                                correct ++;
//                            }
//                        }
//                    }
//
//                    if (count % 1000 == 0) {
//                        double accuracy = 100.0 * correct / total;
//                        System.out.printf("Epoch: %d Loss: %f Accuracy: %f \n", epoch, loss, accuracy);
//                    }
//                }
//            }
//        }


        /*
        int count = 0;
        int numEpochs = 10;
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            var batches = trainLoader.getBatches();
            for (int i = 0; i < batches.size(); ++ i) {
                var trainImages = batches.get(i).images();
                var trainTarget = batches.get(i).labels();

                optimizer.zero_grad();

                var outputs = model.forward(trainImages);

                var loss = error.forward(outputs, trainTarget);

                error.backward(trainImages, trainTarget);

                optimizer.step();

                count += 1;

                if (count % 50 == 0) {
                    var correct = 0;
                    var total = 0;
                    var testBatches = testLoader.getBatches();
                    for (int j = 0; j < testBatches.size(); ++ j) {
                        var testImages = testBatches.get(j).images();
                        var testLabels = testBatches.get(j).labels();

                        var testOutputs = model.forward(testImages);

                        var predicted = getPredictions(testOutputs);

                        total += testLabels.length;

                        for (int k = 0; k < testLabels.length; ++ k) {
                            if (Objects.equals(testLabels[k], predicted[k])) {
                                correct ++;
                            }
                        }
                    }

                    if (count % 1000 == 0) {
                        double accuracy = 100.0 * correct / total;
                        System.out.printf("Epoch: %d Loss: %f Accuracy: %f \n", epoch, loss, accuracy);
                    }
                }
            }
        }
        */
    }

    private static int[] getPredictions(double[][] output) {
        int n = output.length;
        int[] predictions = new int[n];
        for (int i = 0; i < n; ++ i) {
            double maxElement = output[i][0];
            int index = 0;
            for (int j = 0; j < output[i].length; ++ j) {
                if (output[i][j] > maxElement) {
                    maxElement = output[i][j];
                    index = j;
                }
            }
            predictions[i] = index;
        }
        return predictions;
    }


    public double[] flattenImage(double[][] image) {
        int numRows = image.length;
        int numCols = image[0].length;
        double[] flattened = new double[numRows * numCols];

        int index = 0;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                flattened[index++] = image[i][j];
            }
        }

        return flattened;
    }


}
