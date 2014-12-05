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

  public double lambda = .1;
  
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
    W = SimpleMatrix.random(hiddenSize, inputSize + 1, -wInit, wInit, random); 
    U = SimpleMatrix.random(numClasses, hiddenSize + 1, -uInit, uInit, random);
    L = FeatureFactory.allVecs;
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
      SimpleMatrix unbiased = getXForWord(i, word, trainData);   
      SimpleMatrix x = new SimpleMatrix(unbiased.numRows() + 1, unbiased.numCols());
      for (int j = 0; j < unbiased.numRows(); j++) {
        x.set(j, 0, unbiased.get(j, 0));
      }
      x.set(unbiased.numRows(), 0, 1); // bias
 
      //SimpleMatrix p = feedForward(x); 

      gradientCheckReg(x, y); 
    }
	}


  public SimpleMatrix getDelta1(SimpleMatrix delta2) { 
    SimpleMatrix Fz = new SimpleMatrix(z.numRows(), z.numRows()); 
    for (int i=0; i < z.numRows(); i++) {
      double val;
      val = 1 - Math.pow(Math.tanh(z.get(i, 0)), 2); 
      Fz.set(i, i, val);
    }
    // Disregard bias component
    SimpleMatrix unbiasedUTDelta = U.transpose().mult(delta2).extractMatrix(0, hiddenSize, 0, 1); 
    return Fz.mult(unbiasedUTDelta);
  }
  public SimpleMatrix getUGradient(SimpleMatrix delta2) {
    return delta2.mult(h.transpose());  
  }

  public SimpleMatrix getUGradientReg(SimpleMatrix delta2) {
    SimpleMatrix zeroBiasedU = U.copy();
    for (int i = 0; i < numClasses; i++) {
      zeroBiasedU.set(i, hiddenSize, 0);
    }
    return delta2.mult(h.transpose()).plus(zeroBiasedU.scale(lambda));
  }

  public SimpleMatrix getWGradient(SimpleMatrix delta1, SimpleMatrix x) {     
    return delta1.mult(x.transpose());
  }

  public SimpleMatrix getWGradientReg(SimpleMatrix delta1, SimpleMatrix x) {     
    SimpleMatrix zeroBiasedW = W.copy();
    for (int i = 0; i < hiddenSize; i++) {
      zeroBiasedW.set(i, inputSize, 0);
    }
    return delta1.mult(x.transpose()).plus(zeroBiasedW.scale(lambda));
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

    U = U.minus(uGrad.scale(alpha)); 
    W = W.minus(wGrad.scale(alpha));
    // do some manipulation for L update
  } 

  public void gradientCheck(SimpleMatrix x, SimpleMatrix y) { 
    int uSize = U.getNumElements();
    int wSize = W.getNumElements();
    int xSize = x.getNumElements();
    int thetaSize = uSize + wSize + xSize;
    SimpleMatrix pMinus;
    SimpleMatrix pPlus;
    double F;
    for (int i = 0; i < thetaSize; i++) {
      SimpleMatrix p = feedForward(x);
      SimpleMatrix delta2 = p.minus(y);
      SimpleMatrix delta1 = getDelta1(delta2);
      SimpleMatrix UGradient = getUGradient(delta2);
      SimpleMatrix WGradient = getWGradient(delta1, x);
      SimpleMatrix xGradient = getLGradient(delta1);
      if (i < uSize) {
        int m = (i / U.numCols());  
        int n = i % U.numCols();
        U.set(m, n, U.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        U.set(m, n, U.get(m, n) + 2*EPSILON);
        pPlus = feedForward(x); 
        // Reset U back to normal
        U.set(m, n, U.get(m, n) - EPSILON); 
        F = UGradient.get(m, n);
      } else if (i < uSize + wSize && i >= uSize) { 
        int j = i - uSize;
        int n = j % W.numCols();
        int m = (j / W.numCols());  
        W.set(m, n, W.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        W.set(m, n, W.get(m, n) + 2*EPSILON);
        pPlus = feedForward(x); 
        // Reset W back to normal
        W.set(m, n, W.get(m, n) - EPSILON); 
        F = WGradient.get(m, n);
      } else {
        int j = i - (uSize + wSize);
        int n = j % x.numCols();
        int m = (j / x.numCols());  
        x.set(m, n, x.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        x.set(m, n, x.get(m, n) + 2*EPSILON);
        pPlus = feedForward(x); 
        // Reset x back to normal
        x.set(m, n, x.get(m, n) - EPSILON); 
        F = xGradient.get(m, n);
      }
      double JMinus = calcJ(y, pMinus); 
      double JPlus = calcJ(y, pPlus);
      double Jdiff = (JPlus - JMinus)/(2*EPSILON);
      if (Math.abs(F - Jdiff) <= .0000001) {
        //System.out.println("GRADIENT CHECK PASSED");
      } else {
        System.out.println("GRADIENT CHECK FAILED");
        System.out.println(i);
        System.out.println("F: " + F);
        System.out.println("J DIFF: " + Jdiff);
        System.out.println("F&J DIFF: " + Math.abs(F-Jdiff));
      }
    }     
  }

  public void gradientCheckReg(SimpleMatrix x, SimpleMatrix y) { 
    int uSize = U.getNumElements();
    int wSize = W.getNumElements();
    int xSize = x.getNumElements();
    int thetaSize = uSize + wSize + xSize;
    SimpleMatrix pMinus;
    SimpleMatrix pPlus;
    double F;
    double JMinus;
    double JPlus;
    double JDiff;
    for (int i = 0; i < thetaSize; i++) {
      SimpleMatrix p = feedForward(x);
      SimpleMatrix delta2 = p.minus(y);
      SimpleMatrix delta1 = getDelta1(delta2);
      SimpleMatrix UGradient = getUGradientReg(delta2);
      SimpleMatrix WGradient = getWGradientReg(delta1, x);
      SimpleMatrix xGradient = getLGradient(delta1);
      if (i < uSize) {
        int m = (i / U.numCols());  
        int n = i % U.numCols();
        U.set(m, n, U.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        JMinus = calcJReg(y, pMinus); 
        U.set(m, n, U.get(m, n) + 2*EPSILON);
        pPlus = feedForward(x); 
        JPlus = calcJReg(y, pPlus);
        // Reset U back to normal
        U.set(m, n, U.get(m, n) - EPSILON); 
        F = UGradient.get(m, n);
      } else if (i < uSize + wSize && i >= uSize) { 
        int j = i - uSize;
        int n = j % W.numCols();
        int m = (j / W.numCols());  
        W.set(m, n, W.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        JMinus = calcJReg(y, pMinus); 
        W.set(m, n, W.get(m, n) + 2*EPSILON);
        pPlus = feedForward(x); 
        JPlus = calcJReg(y, pPlus);
        // Reset W back to normal
        W.set(m, n, W.get(m, n) - EPSILON); 
        F = WGradient.get(m, n);
      } else {
        int j = i - (uSize + wSize);
        int n = j % x.numCols();
        int m = (j / x.numCols());  
        x.set(m, n, x.get(m, n) - EPSILON); 
        pMinus = feedForward(x);
        JMinus = calcJReg(y, pMinus); 
        x.set(m, n, x.get(m, n) + 2*EPSILON);
        pPlus = feedForward(x); 
        JPlus = calcJReg(y, pPlus);
        // Reset x back to normal
        x.set(m, n, x.get(m, n) - EPSILON); 
        F = xGradient.get(m, n);
      }
      JDiff = (JPlus - JMinus)/(2*EPSILON);
      if (Math.abs(F - JDiff) <= .0000001) {
        //System.out.println("GRADIENT CHECK PASSED");
      } else {
        System.out.println("GRADIENT CHECK FAILED");
        System.out.println(i);
        System.out.println("F: " + F);
        System.out.println("J DIFF: " + JDiff);
        System.out.println("F&J DIFF: " + Math.abs(F-JDiff));
      }
    }     
  }
  
  public double calcJ(SimpleMatrix y, SimpleMatrix p) { 
    for (int i=0; i < y.numRows(); i++) {
      if (y.get(i, 0) == 1.0) { 
        return -Math.log(p.get(i, 0));
      }
    }
    System.out.println("------------THIS IS BAD ----------------");
    return Double.POSITIVE_INFINITY;
  }

  public double calcJReg(SimpleMatrix y, SimpleMatrix p) { 
    return calcJ(y, p) + lambda/2.0 * (Math.pow(W.extractMatrix(0, hiddenSize, 0, inputSize).normF(), 2) +
                                   Math.pow(U.extractMatrix(0, numClasses, 0, hiddenSize).normF(), 2));
  }

  public SimpleMatrix getXForWord(int index, String word, List<Datum> trainData) {
      String wordMinus = trainData.get(index-1).word; 
      String wordPlus = trainData.get(index+1).word;
      System.out.println(wordMinus + ", " + word + ", " + wordPlus);
      int xMinusIndex;
      int xIndex;
      int xPlusIndex;
      if (FeatureFactory.wordToNum.containsKey(wordMinus)) {
        xMinusIndex = FeatureFactory.wordToNum.get(wordMinus); 
      } else {
        xMinusIndex = FeatureFactory.wordToNum.get("UUUNKKK");   
      }
      if (FeatureFactory.wordToNum.containsKey(word)) {
        xIndex = FeatureFactory.wordToNum.get(word); 
      } else {
        xIndex = FeatureFactory.wordToNum.get("UUUNKKK");    
      }
      if (FeatureFactory.wordToNum.containsKey(wordPlus)) {
        xPlusIndex = FeatureFactory.wordToNum.get(wordPlus); 
      } else {
        xPlusIndex = FeatureFactory.wordToNum.get("UUUNKKK");   
      }
      System.out.println(xMinusIndex + ", " + xIndex + ", " + xPlusIndex);
      SimpleMatrix unbiasedWindow = new SimpleMatrix(inputSize, 1);
      for (int i = 0; i < wordSize; i++) { 
        unbiasedWindow.set(i, 0, L.get(xMinusIndex, i));
        unbiasedWindow.set(i+50, 0, L.get(xIndex, i));
        unbiasedWindow.set(i+100, 0, L.get(xPlusIndex, i));
      }
      return unbiasedWindow;
  }


	public void test(List<Datum> testData){
		// TODO
	}

  public SimpleMatrix feedForward(SimpleMatrix window) {
    z = W.mult(window);
    h = new SimpleMatrix(z.numRows() + 1, z.numCols());
    for (int i = 0; i < z.numRows(); i++) {
      h.set(i, 0, Math.tanh(z.get(i, 0)));
    }
    h.set(z.numRows(), 0, 1); // bias
    SimpleMatrix v = U.mult(h);
    return softmax(v); 
  }

  public SimpleMatrix softmax(SimpleMatrix v) {
    double denom = 0.0;    
    double[] numer = new double[v.numRows()];
    for (int i = 0; i < v.numRows(); i++) {
      double expI = Math.exp(v.get(i, 0));
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
