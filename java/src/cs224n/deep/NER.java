package cs224n.deep;

import cs224n.util.FileOutputer; 

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
  
  public static final int H = 100;
  public static final int NUM_ITERS = 10;
  public static final double ALPHA = .01; 
  public static final double LAMBDA = 0; //.00001;


  public static void main(String[] args) throws IOException, Exception {
    if (args.length < 2) {
        System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
        return;
    }	    

    // this reads in the train and test datasets
    List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
    List<Datum> testData = FeatureFactory.readTestData(args[1]);	

    //	read the train and test data
    //TODO: Implement this function (just reads in vocab and word vectors)
    FeatureFactory.initializeVocab("../data/vocab.txt");
    SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors.txt");

    // initialize model 
    WindowModel model = new WindowModel(3, 50, H, 5, ALPHA, LAMBDA, NUM_ITERS);
    model.initWeights();

    //TODO: Implement those two functions
    model.train(trainData);
    List<Datum> predictions = model.test(testData);
    FileOutputer.writePredictionsToFile("../WindowModelPredictions.out", testData, predictions);
  }
}
