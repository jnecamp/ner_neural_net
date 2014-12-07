package cs224n.util;

import cs224n.deep.*;

import java.io.*;
import java.util.*;

import org.ejml.simple.SimpleMatrix;



// Extracts the indices (into L) of words in list and writes indices and given NER tag to file
public class IndexAndLabelWriter {
  
  public static void write(String filename, List<Datum> wordsAndLabels) throws IOException, Exception {
    HashMap<String, Integer> labelToIndex = new HashMap<String, Integer>();
    labelToIndex.put("O", 0);
    labelToIndex.put("LOC", 1);
    labelToIndex.put("MISC", 2);
    labelToIndex.put("ORG", 3);
    labelToIndex.put("PER", 4);
    FeatureFactory.initializeVocab("../data/vocab.txt");
    HashSet<Integer> indicesInVocab = new HashSet<Integer>();
    SimpleMatrix toWrite = new SimpleMatrix(wordsAndLabels.size(), 2);
    int update = 0;
    for (Datum d : wordsAndLabels) {
      Integer index = getLIndexForWord(d.word); 
      if (!indicesInVocab.contains(index)) {
        toWrite.set(update, 0, index);
        System.out.println(d.word + " " + d.label);
        System.out.println(labelToIndex.get(d.label));
        toWrite.set(update, 1, labelToIndex.get(d.label));
        indicesInVocab.add(index);
        update++;
      }
    }
    toWrite = toWrite.extractMatrix(0, indicesInVocab.size(), 0, 2);
    MatrixWriter.write(filename, toWrite);
  }

  private static int getLIndexForWord(String word) { 

    int index = -1;
    word = word.toLowerCase();
    if (FeatureFactory.wordToNum.containsKey(word)) {
      index = FeatureFactory.wordToNum.get(word); 
    } else {
      try {
        Double.parseDouble(word); 
        index = FeatureFactory.wordToNum.get("NNNUMMM");   
      } catch (Exception e) {
        index = FeatureFactory.wordToNum.get("UUUNKKK");   
      }
    }
    return index;
  }

}
