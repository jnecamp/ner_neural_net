package cs224n.util;

import java.io.*;
import java.util.*;

import cs224n.deep.Datum;


public class FileOutputer {


	private FileOutputer() {

	}
	 
	public static void writePredictionsToFile(String filename, List<Datum> gold, List<Datum> predictions) throws IOException, Exception {
    BufferedWriter writer = null;
    try {
          writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(filename), "utf-8"));
          for (int i = 0; i < gold.size(); i++) {
            String word = gold.get(i).word;
            String predictedWord = predictions.get(i).word;
            String goldLabel = gold.get(i).label;
            String predictedLabel = predictions.get(i).label;
            if (!word.equals(predictedWord)) {
              throw new Exception("Y'alls predicted word and gold word don't dun match. You dun fucked up");
            }
            writer.write(word + " " + goldLabel + " " + predictedLabel);
            writer.newLine();

          }
          writer.flush();
    } catch (IOException ex) {
        // report
    } finally {
         try {writer.close();} catch (Exception ex) {}
    }
    
	}
 








}
