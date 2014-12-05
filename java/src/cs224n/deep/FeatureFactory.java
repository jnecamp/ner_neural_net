package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
    Datum startDatum = new Datum("<s>", "O");
    Datum endDatum = new Datum("</s>", "O");
    data.add(startDatum);
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
        data.add(endDatum);
        data.add(startDatum);
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}
    data.add(endDatum);
		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
    allVecs = new SimpleMatrix(100232, 50);
    BufferedReader in = new BufferedReader(new FileReader(vecFilename));   
    int m = 0;
    for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}	    	
      String[] strArr = line.split(" ");   
      for (int n = 0; n < strArr.length; n++) {
        String strVal = strArr[n];
        double val = Double.parseDouble(strVal); 
        allVecs.set(m, n, val);
      }
      m++;
    } 
    return allVecs;
	}
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
    BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
    int index = 0;
    for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}	    	
      wordToNum.put(line, index); 
      numToWord.put(index, line);
      index++;
    }
		return wordToNum;
	}
}
