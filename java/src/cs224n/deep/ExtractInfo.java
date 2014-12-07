package cs224n.deep;

import cs224n.util.*; 

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class ExtractInfo {

  public static void main(String[] args) throws IOException, Exception {

    // this reads in the train and test datasets
    List<Datum> trainData = FeatureFactory.readTrainData(args[0]);

    IndexAndLabelWriter.write("../tSNE_matlab/trainData_indeces_and_labels.txt", trainData);

  }
}
