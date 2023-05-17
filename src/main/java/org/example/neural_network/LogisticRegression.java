package org.example.neural_network;


import java.util.ArrayList;
import java.util.List;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LogisticRegression {
    private double[][] weights;
    private double learningRate;
    private int numIterations;

    public LogisticRegression(int numFeatures, double learningRate, int numIterations) {
        this.learningRate = learningRate;
        this.numIterations = numIterations;

        // Initialize weights randomly
        Random random = new Random();
        weights = new double[10][numFeatures];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < numFeatures; j++) {
                weights[i][j] = random.nextDouble();
            }
        }
    }

    public void train(List<double[]> X, List<Integer> y) {
        int numInstances = X.size();
        int numFeatures = X.get(0).length;

        for (int iteration = 0; iteration < numIterations; iteration++) {
            double[][] gradients = new double[10][numFeatures];

            for (int i = 0; i < numInstances; i++) {
                double[] instance = X.get(i);
                int label = y.get(i);

                double[] scores = new double[10];

                // Calculate scores
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < numFeatures; k++) {
                        scores[j] += weights[j][k] * instance[k];
                    }
                }

                // Calculate probabilities using softmax
                double[] probabilities = softmax(scores);

                // Calculate the gradients
                for (int j = 0; j < 10; j++) {
                    double gradient = probabilities[j];
                    if (j == label) {
                        gradient -= 1.0;
                    }

                    for (int k = 0; k < numFeatures; k++) {
                        gradients[j][k] += gradient * instance[k];
                    }
                }
            }

            // Update weights
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    weights[i][j] -= learningRate * gradients[i][j] / numInstances;
                }
            }

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

        }
    }

    public int predict(double[] X) {
        int numFeatures = X.length;
        double[] scores = new double[10];

        // Calculate scores
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < numFeatures; j++) {
                scores[i] += weights[i][j] * X[j];
            }
        }

        // Calculate probabilities using softmax
        double[] probabilities = softmax(scores);

        int maxIndex = 0;
        double maxProbability = probabilities[0];

        // Find the index with the highest probability
        for (int i = 1; i < 10; i++) {
            if (probabilities[i] > maxProbability) {
                maxIndex = i;
                maxProbability = probabilities[i];
            }
        }

        return maxIndex;
    }

    private double[] softmax(double[] scores) {
        double[] probabilities = new double[scores.length];
        double maxScore = Double.NEGATIVE_INFINITY;
        double sum = 0.0;

        for (double score : scores) {
            if (score > maxScore) {
                maxScore = score;
            }
        }

        // Compute the softmax probabilities
        for (int i = 0; i < scores.length; i++) {
            probabilities[i] = Math.exp(scores[i] - maxScore);
            sum += probabilities[i];
        }

        // Normalize the probabilities
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    public double calculateLoss(List<double[]> X, List<Integer> y) {
        int numInstances = X.size();
        int numFeatures = X.get(0).length;
        double loss = 0.0;

        for (int i = 0; i < numInstances; i++) {
            double[] instance = X.get(i);
            int label = y.get(i);

            double[] scores = new double[10];

            // Calculate scores
            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < numFeatures; k++) {
                    scores[j] += weights[j][k] * instance[k];
                }
            }

            // Calculate probabilities using softmax
            double[] probabilities = softmax(scores);

            // Calculate the cross-entropy loss
            double correctProbability = probabilities[label];
            loss += -Math.log(correctProbability);
        }

        loss /= numInstances;

        return loss;
    }


}