import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import Jama.Matrix;
import dataset.Attribute;
import dataset.Dataset;
import parameterTuning.plotting.MathematicaListCreator;
import utilities.DoubleCompare;
import utilities.StopWatch;

public class LinearRegressor {
	public Matrix trainX;
	public Matrix trainY;
	public Matrix testX;
	public Matrix testY;
	public Matrix trainXPseduoInverse;
	public Matrix weightsWithoutRegularization;
	public int[] colunIndices;
	public Dataset dataset;
	LinearRegressor(Dataset dataset) {
		this.dataset = dataset;
		trainX = convertAttributeArrayToMatrix(dataset.getTrainingInstances());
		trainY = convertAttributeArrayToMatrix(dataset.getTrainingResponses());
		testX = convertAttributeArrayToMatrix(dataset.getTestInstances());
		testY = convertAttributeArrayToMatrix(dataset.getTestResponses());
		//trainXPseduoInverse = trainX.inverse();
		//weightsWithoutRegularization = trainXPseduoInverse.times(trainY);
		//double trainingError = getRMSE(trainX, trainY, weightsWithoutRegularization);
		//double testError = getRMSE(testX, testY, weightsWithoutRegularization);
		//System.out.println(String.format("Training RMSE: %f\nTest RMSE: %f", trainingError, testError));
		//generateErrorsByExampleCountCurve();
		generateErrorsByExampleCountCurveMultiThread();
	}
	
	public static Matrix getOptimalWeightsWithoutRegularization(Matrix X, Matrix Y) {
		return X.inverse().times(Y);
	}
	
	public Matrix convertAttributeArrayToMatrix(Attribute[][] array) {
		double[][] doubleArray = new double[array.length][array[0].length+1];
		for (int i = 0; i < array.length; i++) {
			doubleArray[i][0] = 1;
			for (int j = 1; j < array[0].length+1; j++) {
				try {
					if (!array[i][j-1].isMissingValue()) {
						doubleArray[i][j] = array[i][j-1].getNumericValue();
					}
				} catch (Exception e) {
					System.out.println();
				}
			}
		}
		return new Matrix(doubleArray);
	}
	
	public Matrix convertAttributeArrayToMatrix(Attribute[] array) {
		double[][] doubleArray = new double[array.length][1];
		for (int i = 0; i < array.length; i++) {
			doubleArray[i][0] = array[i].getNumericValue();
		}
		return new Matrix(doubleArray);
	}
	
	public void printDimensionsOfStuff() {
		System.out.println("TrainX:" + trainX.getRowDimension() + " " + trainX.getColumnDimension());
		System.out.println("TrainY:" + trainY.getRowDimension() + " " + trainY.getColumnDimension());
		System.out.println("W:" + weightsWithoutRegularization.getRowDimension() + " " + weightsWithoutRegularization.getColumnDimension());
		//System.out.println("Error:" + error.getRowDimension() + " " + error.getColumnDimension());
		weightsWithoutRegularization.print(4, 2);;
	}
	
	public double getTrainingRMSEWithoutRegularization(Matrix X, Matrix Y, Matrix w, int numberOfExamples) {
		Matrix residuals = X.times(w).minusEquals(Y);
		return Math.sqrt(residuals.transpose().times(residuals).get(0, 0) / trainX.getRowDimension());
	}
	
	public static double getRMSE(Matrix X, Matrix Y, Matrix w) {
		Matrix residuals = X.times(w).minusEquals(Y);
		return Math.sqrt(residuals.transpose().times(residuals).get(0, 0) / X.getRowDimension());
	}
	
	public double sumOfSquares(Matrix m) {
		if (m.getColumnDimension() != 1) {
			System.out.println("SumOFSquares only defined on vectors");
		}
		double sumOfSquares = 0.0;
		double[] elements = m.getColumnPackedCopy();
		for (int i = 0; i < elements.length; i++) {
			sumOfSquares += elements[i] * elements[i];
		}
		return sumOfSquares;
	}
	
	private static class ExampleCountCallable implements Callable<Void>{
		int firstExampleIndex = 0, lastExampleIndex = 0;
		public static double[] trainingError;
		public static double[] testError;
		public static Matrix trainX, trainY, testX, testY;
		
		public ExampleCountCallable(int firstExampleIndex, int lastExampleIndex) {
			this.firstExampleIndex = firstExampleIndex;
			this.lastExampleIndex = lastExampleIndex;
		}
		@Override
		public Void call() throws Exception {
			StopWatch timer = new StopWatch().start();
			for (int i = firstExampleIndex; i <= lastExampleIndex; i+=50) {
				StopWatch timer2 = new StopWatch().start();
				Matrix X = trainX.getMatrix(0, i, 0, trainX.getColumnDimension()-1);
				Matrix Y = trainY.getMatrix(0, i, 0, trainY.getColumnDimension()-1);
				Matrix w = getOptimalWeightsWithoutRegularization(X, Y);
				trainingError[i] = getRMSE(X, Y, w);
				testError[i] = getRMSE(testX, testY, w);
				if (  (firstExampleIndex - i) % 100 == 0) {
					System.out.println(String.format("Started at %d, currently at %d, going to %d. Spent %.4f minutes so far", firstExampleIndex, i, lastExampleIndex, timer.getElapsedMinutes()));
				}
			}
			
			return null;
		}
		
	}
	public void generateErrorsByExampleCountCurve() {
		StopWatch globalTimer = new StopWatch().start();
		StopWatch timer = new StopWatch().start();
		double[] trainingError = new double[trainX.getRowDimension()];
		double[] testError = new double[trainX.getRowDimension()];
		double minTestRMSE = Double.MAX_VALUE, maxRMSE = Double.MIN_VALUE;
		int bestNumberOfExamples = 0;
		for (int lastExampleIndex = trainX.getColumnDimension()-1; lastExampleIndex < trainX.getRowDimension(); lastExampleIndex++) {
			timer.start();
			Matrix X = trainX.getMatrix(0, lastExampleIndex, 0, trainX.getColumnDimension()-1);
			Matrix Y = trainY.getMatrix(0, lastExampleIndex, 0, trainY.getColumnDimension()-1);
			Matrix w = getOptimalWeightsWithoutRegularization(X, Y);
			//w.print(4, 2);
			trainingError[lastExampleIndex] = getRMSE(X, Y, w);
			testError[lastExampleIndex] = getRMSE(testX, testY, w);
			if (DoubleCompare.greaterThan(trainingError[lastExampleIndex], maxRMSE)) {
				maxRMSE = trainingError[lastExampleIndex];
			}
			if (DoubleCompare.greaterThan(testError[lastExampleIndex], maxRMSE)) {
				maxRMSE = testError[lastExampleIndex];
			}
			if (DoubleCompare.lessThan(testError[lastExampleIndex], minTestRMSE)) {
				minTestRMSE = testError[lastExampleIndex];
				bestNumberOfExamples = lastExampleIndex;
			}
			System.out.println(String.format("One iteration in %f: ", timer.getElapsedSeconds()));
		}
		
		for (int i = 0; i < trainX.getColumnDimension()-1; i++) {
			trainingError[i] = trainingError[trainX.getColumnDimension()-1];
			testError[i] = testError[trainX.getColumnDimension()-1];
		}
		System.out.println(String.format("All iterations in %f: ", globalTimer.getElapsedSeconds()));
		try {
			BufferedWriter mathematica = new BufferedWriter(new PrintWriter(new File(dataset.parameters.minimalName + "--errorsByExampleCountCurve.txt")));
			mathematica.write("trainingError := " + MathematicaListCreator.convertToMathematicaList(trainingError) + "\n");
			mathematica.write("testError := " + MathematicaListCreator.convertToMathematicaList(testError) + "\n");
			mathematica.write("optimalNumberOfExamples := {{" + bestNumberOfExamples + ", 0}, {" + bestNumberOfExamples + ", " + maxRMSE + "}}\n");
			mathematica.write("errorsByExampleCountCurve := ListLinePlot[{trainingError,testError, optimalNumberOfExamples}"
					+ ", PlotLegends -> {\"trainingError\", \"testError\", \"optimalNumberOfExamples\"}"
					+ ", PlotStyle -> {Blue, Orange, Green}"
					+ ", AxesLabel->{\"Number Of Training Examples\", \"RMSE\"}"
					+ ", PlotRange -> {{Automatic, Automatic}, {0, " + maxRMSE + "}}"
					+ "] \nerrorsByExampleCountCurve\n\n");
			//mathematica.write(saveToFile.toString());
			//mathematica.write(latexCode.toString());
			mathematica.flush();
			mathematica.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public ExecutorService executor = Executors.newCachedThreadPool();
	public void generateErrorsByExampleCountCurveMultiThread() {
		StopWatch globalTimer = new StopWatch().start();
		ExampleCountCallable.trainingError = new double[trainX.getRowDimension()];
		ExampleCountCallable.testError = new double[trainX.getRowDimension()];
		ExampleCountCallable.trainX = trainX;
		ExampleCountCallable.trainY = trainY;
		ExampleCountCallable.testX = testX;
		ExampleCountCallable.testY = testY;
		double minTestRMSE = Double.MAX_VALUE, maxRMSE = Double.MIN_VALUE;
		int bestNumberOfExamples = 0;
		ArrayList<Future<Void>> futures = new ArrayList<>();
		
		int inEachCallable = (int)Math.ceil(trainX.getRowDimension() / 2);
		for (int lastExampleIndex = trainX.getColumnDimension()-1; 
				 lastExampleIndex < trainX.getRowDimension();
				 lastExampleIndex += inEachCallable+1) {
			futures.add(executor.submit(new ExampleCountCallable(lastExampleIndex, Math.min(lastExampleIndex+inEachCallable, trainX.getRowDimension()))));
			inEachCallable/=1.5;
			System.out.println(String.format("Started one from %d to %d: ",lastExampleIndex, Math.min(lastExampleIndex+inEachCallable, trainX.getRowDimension())));
		}
		System.out.println(String.format("Submitted all callables in %f: ", globalTimer.getElapsedSeconds()));

		StopWatch timer = new StopWatch();
		for (Future<Void> future : futures) {
			timer.start(); 
			try {
				future.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			System.out.println(String.format("One iteration in %f: ", timer.getElapsedSeconds()));
		}
		for (int i = 0; i < ExampleCountCallable.trainingError.length; i++) {
			if (DoubleCompare.greaterThan(ExampleCountCallable.trainingError[i], maxRMSE)) {
				maxRMSE = ExampleCountCallable.trainingError[i];
			}
			if (DoubleCompare.greaterThan(ExampleCountCallable.testError[i], maxRMSE)) {
				maxRMSE = ExampleCountCallable.testError[i];
			}
			if (DoubleCompare.lessThan(ExampleCountCallable.testError[i], minTestRMSE)) {
				minTestRMSE = ExampleCountCallable.testError[i];
				bestNumberOfExamples = i;
			}
		}
		for (int i = 0; i < trainX.getColumnDimension()-1; i++) {
			ExampleCountCallable.trainingError[i] = ExampleCountCallable.trainingError[trainX.getColumnDimension()-1];
			 ExampleCountCallable.testError[i] = ExampleCountCallable.testError[trainX.getColumnDimension()-1];
		}

		System.out.println(String.format("All iterations in %f: ", globalTimer.getElapsedSeconds()));
		try {
			BufferedWriter mathematica = new BufferedWriter(new PrintWriter(new File(dataset.parameters.minimalName + "--errorsByExampleCountCurveMultiThread.txt")));
			mathematica.write("trainingError := " + MathematicaListCreator.convertToMathematicaList(ExampleCountCallable.trainingError) + "\n");
			mathematica.write("testError := " + MathematicaListCreator.convertToMathematicaList(ExampleCountCallable.testError) + "\n");
			mathematica.write("optimalNumberOfExamples := {{" + bestNumberOfExamples + ", 0}, {" + bestNumberOfExamples + ", " + maxRMSE + "}}\n");
			mathematica.write("errorsByExampleCountCurve := ListLinePlot[{trainingError,testError, optimalNumberOfExamples}"
					+ ", PlotLegends -> {\"trainingError\", \"testError\", \"optimalNumberOfExamples\"}"
					+ ", PlotStyle -> {Blue, Orange, Green}"
					+ ", AxesLabel->{\"Number Of Training Examples\", \"RMSE\"}"
					+ ", PlotRange -> {{Automatic, Automatic}, {0, " + maxRMSE + "}}"
					+ "] \nerrorsByExampleCountCurve\n\n");
			//mathematica.write(saveToFile.toString());
			//mathematica.write(latexCode.toString());
			mathematica.flush();
			mathematica.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
