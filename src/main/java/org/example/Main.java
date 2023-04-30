package org.example;

import org.example.entity.data.DataLoader;
import org.example.entity.data.Dataset;
import org.example.entity.torch.Linear;
import org.example.evaluation.CrossEntropyLoss;
import org.example.neural_network.LogisticRegression;
import org.example.entity.torch.SGD;
import org.example.preprocessing.PreprocessService;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;


public class Main {
    public static void main(String[] args) {
        Dataset train = new Dataset("src/main/java/org/example/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();
//        System.out.println(train.headTrainY());

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
        LogisticRegression model = new LogisticRegression(inputDim, outputDim);
        Linear linear = new Linear(inputDim, outputDim);

        CrossEntropyLoss error = new CrossEntropyLoss(10);

        double learningRate = 0.01;
        SGD optimizer = new SGD(linear, learningRate);

        int count = 0;
        int numEpochs = 10;
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            var batches = trainLoader.getBatches();
            for (int i = 0; i < batches.size(); ++ i) {
                var trainImages = batches.get(i).getImages();
                var trainTarget = batches.get(i).getLabels().stream().mapToInt(Integer::intValue).toArray();

                optimizer.zero_grad();

                var outputs = model.forward(convertToDoubleArray(trainImages));

                var loss = error.forward(outputs, trainTarget);

                error.backward(convertToDoubleArray(trainImages), trainTarget);

                optimizer.step();

                count += 1;

                if (count % 50 == 0) {
                    var correct = 0;
                    var total = 0;
                    var testBatches = testLoader.getBatches();
                    for (int j = 0; j < testBatches.size(); ++ j) {
                        var testImages = testBatches.get(j).getImages();
                        var testLabels = testBatches.get(j).getLabels();

                        var testOutputs = linear.forward(convertToDoubleArray(testImages));

                        var predicted = getPredictions(outputs);

                        total += testLabels.size();

                        for (int k = 0; k < testLabels.size(); ++ k) {
                            if (Objects.equals(testLabels.get(k), predicted.get(k))) {
                                correct ++;
                            }
                        }

                        double accuracy = 100.0 * correct / total;

                    }

                    if (count % 500 == 0) {
                        System.out.printf("Epoch: %d Loss: %f\n", epoch, loss);
//                            System.out.printf("Iteration: {%d}  Accuracy: {%f}\n", count, accuracy);
                    }
                }
            }
        }
    }

    private static List<Integer> getPredictions(double[][] output) {
        List<Integer> predictions = new ArrayList<>();
        for (int i = 0; i < output.length; ++ i) {
            double maxElement = output[i][0];
            int index = 0;
            for (int j = 0; j < output[i].length; ++ j) {
                if (output[i][j] > maxElement) {
                    maxElement = output[i][j];
                    index = j;
                }
            }
            predictions.add(index);
        }
        return predictions;
    }

    private static double[][] convertToDoubleArray(List<List<Double>> train_x) {
        return train_x.stream()
                .map(l -> l.stream().mapToDouble(Double::doubleValue).toArray())
                .toArray(double[][]::new);
    }
}
