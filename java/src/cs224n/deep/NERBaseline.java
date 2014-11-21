package cs224n.deep;

import cs224n.util.FileOutputer; 

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NERBaseline {
    
  public static void main(String[] args) throws IOException, Exception {
    if (args.length < 2) {
        System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
        return;
    }	    

    // this reads in the train and test datasets
    List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
    List<Datum> testData = FeatureFactory.readTestData(args[1]);	

    train(trainData);
    List<Datum> predictions = test(testData);

    FileOutputer.writePredictionsToFile("../BaselinePredictions.out", testData, predictions);
  }

  private static HashMap<String, String> exactMatches;

  private static void train(List<Datum> trainData) {
    exactMatches = new HashMap<String, String>();
    for (Datum d : trainData) {
      exactMatches.put(d.word, d.label);
    }
  }

  private static List<Datum> test(List<Datum> testData) {
    List<Datum> predictions = new ArrayList<Datum>();
    for (Datum d : testData) {
      String predictedLabel = exactMatches.get(d.word);
      if (predictedLabel == null) {
        predictedLabel = "O";
      }
      predictions.add(new Datum(d.word, predictedLabel));
    }
    return predictions;
  }

}
