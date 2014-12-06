package cs224n.deep;

import cs224n.util.FileOutputer; 

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NERValidate {

  public static ArrayList<Double> LAMBDAS, ALPHAS;
  public static ArrayList<Integer> ITERS, H;

  public static void main(String[] args) throws IOException, Exception {
    if (args.length < 2) {
        System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
        return;
    }	    
    LAMBDAS = new ArrayList<Double>();
    LAMBDAS.add(.001); 
    LAMBDAS.add(.0001); 
    LAMBDAS.add(.00001); 
    LAMBDAS.add(.000001); 

    ALPHAS = new ArrayList<Double>();
    ALPHAS.add(.1); 
    ALPHAS.add(.01); 
    ALPHAS.add(.001); 
    ALPHAS.add(.0001); 
    ALPHAS.add(.00001); 
    ALPHAS.add(.000001); 

    ITERS = new ArrayList<Integer>();
    ITERS.add(2); 
    ITERS.add(4); 
    ITERS.add(8); 
    ITERS.add(16); 
    
    H = new ArrayList<Integer>();
    H.add(75); 
    H.add(100); 
    H.add(150); 
    H.add(200);

    // this reads in the train and test datasets
    List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
    List<Datum> testData = FeatureFactory.readTestData(args[1]);	

    //	read the train and test data
    //TODO: Implement this function (just reads in vocab and word vectors)
    FeatureFactory.initializeVocab("../data/vocab.txt");
    for (double lambda : LAMBDAS) {
      for (double alpha : ALPHAS) { 
        for (int numIters : ITERS) { 
          for (int hiddenSize : H) { 
            SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors.txt");

            // initialize model 
      
            WindowModel model = new WindowModel(3, 50, hiddenSize, 5, alpha, lambda, numIters);
            model.initWeights();

            //TODO: Implement those two functions
            model.train(trainData);
            List<Datum> predictions = model.test(testData);
            String filename = "window_" + lambda + "_" + alpha + "_" + numIters + "_" + hiddenSize + ".out";
            FileOutputer.writePredictionsToFile("../" + filename, testData, predictions);
          }
        }
      }
    }
  }
}
