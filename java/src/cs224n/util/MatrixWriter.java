package cs224n.util;

import java.io.*;
import java.util.*;

import org.ejml.simple.SimpleMatrix;


public class MatrixWriter {


	private MatrixWriter() {

	}
	 
  // Writes the given matrix to file.  If to be used in matlab, make sure to delete the first
  // line that gets written.
	public static void write(String filename, SimpleMatrix toWrite) throws IOException, Exception {
    BufferedWriter writer = null;
    try {
      writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(filename), "utf-8"));
      writer.write(toWrite.toString());
      writer.flush();
    } catch (IOException ex) {
        // report
    } finally {
       try {writer.close();} catch (Exception ex) {}
    }
    
	}
}
