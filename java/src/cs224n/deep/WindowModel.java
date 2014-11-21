package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, U;
	//
	public int windowSize,wordSize, hiddenSize;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		//TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		//	TODO
	}

	
	public void test(List<Datum> testData){
		// TODO
	}

  public double[] feed_forward(SimpleMatrix unbiasedWindow) {
    SimpleMatrix window = new SimpleMatrix(unbiasedWindow.numRows() + 1, unbiasedWindow.numCols());
    for (int i = 0; i < unbiasedWindow.numRows(); i++) {
      window.set(i, 1, window.get(i, 1));
    }
    window.set(unbiasedWindow.numRows(), 1, 1); // bias
    SimpleMatrix z = W.mult(window);
    SimpleMatrix h = new SimpleMatrix(z.numRows() + 1, z.numCols());
    for (int i = 0; i < z.numRows(); i++) {
      h.set(i, 1, Math.tanh(z.get(i, 1)));
    }
    h.set(z.numRows(), 1); // bias 
    SimpleMatrix v = U.mult(h);
    return softmax(v); 
  }

  public double[] softmax(SimpleMatrix v) {
    double denom = 0.0;    
    double[] numer = new double[v.numRows()];
    for (int i = 0; i < v.numRows(); i++) {
      double expI = Math.exp(v.get(i, 1));
      denom += expI; 
      numer[i] = expI;
    }
    for (int i = 0; i < numer.length; i++) {
      numer[i] = numer[i]/denom;
    }
    return numer;
  }
	
}
