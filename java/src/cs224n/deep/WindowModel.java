package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, U;

	public int windowSize, wordSize, numClasses, hiddenSize, inputSize;
  
  public double alpha;

	public WindowModel(int _windowSize, int _wordSize, int _hiddenSize, int _classes, double _lr){
    windowSize = _windowSize; 
    wordSize = _wordSize; 
    hiddenSize = _hiddenSize;
    numClasses = _classes;
    alpha = _lr; 
    inputSize = windowSize * wordSize; 
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
    Random random = new Random();
    double wInit = Math.sqrt(6)/Math.sqrt(inputSize + hiddenSize);
    double uInit = Math.sqrt(6)/Math.sqrt(hiddenSize + numClasses);
    SimpleMatrix W = SimpleMatrix.random(hiddenSize, inputSize, -wInit, wInit, random); 
    SimpleMatrix U = SimpleMatrix.random(numClasses, hiddenSize, -uInit, uInit, random);
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
