package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, U, h, z;

  public final double EPSILON = .0001;

	public int windowSize, wordSize, numClasses, hiddenSize, inputSize;
  
  public double alpha;
  
  public HashMap<String, Integer> labelToIndex;

	public WindowModel(int _windowSize, int _wordSize, int _hiddenSize, int _classes, double _lr){
    windowSize = _windowSize; 
    wordSize = _wordSize; 
    hiddenSize = _hiddenSize;
    numClasses = _classes;
    alpha = _lr; 
    inputSize = windowSize * wordSize; 
    labelToIndex = new HashMap<String, Integer>();
    labelToIndex.put("O", 0);
    labelToIndex.put("LOC", 1);
    labelToIndex.put("MISC", 2);
    labelToIndex.put("ORG", 3);
    labelToIndex.put("PER", 4);
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
    SimpleMatrix L = FeatureFactory.allVecs;
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		//	TODO shuffle the data for SGD 
    List<Datum> trainData = _trainData;
    for (int i = 1; i < trainData.size()-1; i++) {
      Datum datum = trainData.get(i);
      String word = datum.word;
      if (word == "<s>" || word == "</s>") {
        continue;
      }

      SimpleMatrix y = new SimpleMatrix(numClasses, 1);  
      y.set(labelToIndex.get(datum.label), 1);  
      SimpleMatrix x = getXForWord(i, word, trainData);   

      SimpleMatrix p = feedForward(x); 
       
    }
	}

  public SimpleMatrix getUGradient(SimpleMatrix delta2) {
    return delta2.mult(h.transpose());  
  }

  public SimpleMatrix getDelta1(SimpleMatrix delta2) { 
    SimpleMatrix Fz = new SimpleMatrix(z.numRows(), z.numRows()); 
    for (int i=0; i < z.numRows(); i++) {
      double val = 1 - Math.pow(Math.tanh(z.get(i, 1)), 2); 
      Fz.set(i, i, val);
    }
    return Fz.mult(U.transpose()).mult(delta2);
  }

  public SimpleMatrix getWGradient(SimpleMatrix delta1, SimpleMatrix x) {     
    return delta1.mult(x.transpose());
  }

  public SimpleMatrix getLGradient(SimpleMatrix delta1) { 
    return W.transpose().mult(delta1);
  }

  public void backprop(SimpleMatrix x, SimpleMatrix p, SimpleMatrix y){
    SimpleMatrix delta2 = p.minus(y);
    SimpleMatrix delta1 = getDelta1(delta2);

    SimpleMatrix uGrad = getUGradient(delta2); 
    SimpleMatrix wGrad = getWGradient(delta1, x); 
    SimpleMatrix lGrad = getLGradient(delta1);
  } 

  public void gradientCheck(SimpleMatrix x, SimpleMatrix y) { 
    SimpleMatrix oldU = U;
    SimpleMatrix oldW = W;
    SimpleMatrix oldL = L;
    int uSize = U.getNumElements();
    int wSize = W.getNumElements();
    int lSize = L.getNumElements();
    int thetaSize = uSize + wSize + lSize;
    SimpleMatrix pMinus;
    SimpleMatrix pPlus;
    double F;
    for (int i = 0; i < thetaSize; i++) {
      if (i < uSize) {
        int m = (i / U.numCols());  
        int n = i % U.numCols();
        U.set(m, n, oldU.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        U.set(m, n, oldU.get(m, n) + EPSILON);
        pPlus = feedForward(x); 
        U = oldU;
        SimpleMatrix p = feedForward(x);
        SimpleMatrix delta2 = p.minus(y);
        F = getUGradient(delta2).get(m, n);
      } else if (i < uSize + wSize && i > uSize) { 
        int j = i - uSize;
        int n = j % W.numCols();
        int m = (j / W.numCols());  
        W.set(m, n, oldW.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        W.set(m, n, oldW.get(m, n) + EPSILON);
        pPlus = feedForward(x); 
        W = oldW;
        SimpleMatrix p = feedForward(x);
        SimpleMatrix delta2 = p.minus(y);
        SimpleMatrix delta1 = getDelta1(delta2);
        F = getWGradient(delta1, x).get(m, n);
      } else {
        int j = i - (uSize + wSize);
        int n = j % L.numCols();
        int m = (j / L.numCols());  
        L.set(m, n, oldL.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        L.set(m, n, oldL.get(m, n) + EPSILON);
        pPlus = feedForward(x); 
        L = oldL;
        SimpleMatrix p = feedForward(x);
        SimpleMatrix delta2 = p.minus(y);
        SimpleMatrix delta1 = getDelta1(delta2);
        F = getLGradient(delta1).get(m, n);
      }
      double JMinus = calcJ(y, pMinus); 
      double JPlus = calcJ(y, pPlus);
      double Jdiff = (JPlus - JMinus)/(2*EPSILON);
      if (Math.abs(F - Jdiff) <= .0000007) {
        System.out.println("GRADIENT CHECK PASSED");
      } else {
        System.out.println("GRADIENT CHECK FAILED");
      }
    }     
  }
  
  public double calcJ(SimpleMatrix y, SimpleMatrix p) { 
    for (int i=0; i < y.numRows(); i++) {
      if (y.get(i, 1) == 1.0) { 
        return Math.log(p.get(i, 1));
      }
    }
    return Double.POSITIVE_INFINITY;
  }

  public SimpleMatrix getXForWord(int index, String word, List<Datum> trainData) {
      String wordMinus = trainData.get(index-1).word; 
      String wordPlus = trainData.get(index+1).word;

      int xMinusIndex = FeatureFactory.wordToNum.get(wordMinus); 
      int xIndex = FeatureFactory.wordToNum.get(word); 
      int xPlusIndex = FeatureFactory.wordToNum.get(wordPlus);

      SimpleMatrix unbiasedWindow = new SimpleMatrix(inputSize, 1);
      for (int i = 0; i < windowSize; i++) { 
        unbiasedWindow.set(i, 1, L.get(xMinusIndex, i));
        unbiasedWindow.set(i+50, 1, L.get(xIndex, i));
        unbiasedWindow.set(i+100, 1, L.get(xIndex, i));
      }
      return unbiasedWindow;
  }


	public void test(List<Datum> testData){
		// TODO
	}

  public SimpleMatrix feedForward(SimpleMatrix unbiasedWindow) {
    SimpleMatrix window = new SimpleMatrix(unbiasedWindow.numRows() + 1, unbiasedWindow.numCols());
    for (int i = 0; i < unbiasedWindow.numRows(); i++) {
      window.set(i, 1, window.get(i, 1));
    }
    window.set(unbiasedWindow.numRows(), 1, 1); // bias
    z = W.mult(window);
    h = new SimpleMatrix(z.numRows() + 1, z.numCols());
    for (int i = 0; i < z.numRows(); i++) {
      h.set(i, 1, Math.tanh(z.get(i, 1)));
    }
    h.set(z.numRows(), 1); // bias 
    SimpleMatrix v = U.mult(h);
    return softmax(v); 
  }

  public SimpleMatrix softmax(SimpleMatrix v) {
    double denom = 0.0;    
    double[] numer = new double[v.numRows()];
    for (int i = 0; i < v.numRows(); i++) {
      double expI = Math.exp(v.get(i, 1));
      denom += expI; 
      numer[i] = expI;
    }
    SimpleMatrix p = new SimpleMatrix(v.numRows(), 1);
    for (int i = 0; i < numer.length; i++) {
       p.set(i, numer[i]/denom);
    }
    return p;
  }
	
}
