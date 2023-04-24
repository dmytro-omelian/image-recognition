package org.example;

import org.example.entity.data.DataLoader;
import org.example.entity.data.Dataset;
import org.example.entity.torch.Linear;
import org.example.entity.torch.Loss;
import org.example.neural_network.LogisticRegression;
import org.example.entity.torch.SGD;
import org.example.preprocessing.PreprocessService;


public class Main {
    public static void main(String[] args) {
        Dataset train = new Dataset("src/main/java/org/example/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();
        System.out.println(train.headTrainY());

        // FIXME features -> it is double values (especially after normalization)

        var features = train.getNormalizedFeatures();
        var labels = train.getLabels();

        PreprocessService.TrainTestEntity trainTestEntity = PreprocessService.trainTestSplit(
                train.getFeatures(), train.getLabels(), 0.2);

        var train_X = trainTestEntity.getTrainX();
        var train_y = trainTestEntity.getTrainY();
        var test_X = trainTestEntity.getTestX();
        var test_y = trainTestEntity.getTestY();

        var trainLoader = new DataLoader(train_X, train_y, 8);
        var testLoader = new DataLoader(test_X, test_y, 8);

        // visualize image

        int inputDim = 28 * 28;
        int outputDim = 10;
        LogisticRegression model = new LogisticRegression(inputDim, outputDim);

//        var error = new CrossEntropyLoss();
//        var optimizer = new SGD();
//
//        int numberOfEpochs = 10; // FIXME
//
//        for (int epoch = 0; epoch < numberOfEpochs; ++epoch) {
//
//            for (var batch : trainLoader.getBatches()) {
//
////                var images = batch.
//
////                optimizer.zero_grad();
//
////                var outputs = model.forward();
//
////                var loss = error.calculate(outputs, targets)
//
////                optimizer.step();
//            }
//
//        }
        double[][] input = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}};
        double[][] target = {{0.3, 0.5}, {0.7, 0.2}, {0.1, 0.8}};

        Linear linear = new Linear(3, 2);

        Loss loss = new Loss();

        double learningRate = 0.01;
        SGD optimizer = new SGD(linear, learningRate);

        int numEpochs = 1000;
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            double[][] output = linear.forward(input);
            double[][] gradOutput = loss.backward(output, target);
            double[][] gradInput = linear.backward(input, gradOutput, learningRate);
            optimizer.step();

            if (epoch % 100 == 0) {
                double lossValue = loss.forward(output, target);
                System.out.printf("Epoch %d: Loss = %.4f\n", epoch, lossValue);
            }
        }



        // for each epoch
        // for each batch
        // convert train and labels to smth
        // optimizer.zero_grad()
        // calculate model outputs // forward propagation
        // calculate loss with CrossEntropyLoss
        // loss backward()
        // optimizer.step()
        // count += 1
        // calculate prediction
        // - print loss

    }
}
