import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.Callable;

import utilities.DoubleCompare;
import utilities.MersenneTwisterFast;
import utilities.SimpleHostLock;
import utilities.StopWatch;
import Jama.Matrix;
enum UpdateRule {Original, AdaptedLR}
public class LinearRegressor {
	
	public LinearRegressorDataset dataset;
	public Matrix trainXSquaredDivideOneHalfN;
	public Matrix trainXTransposeTimesTrainYDivideOneHalfN;
	
	public Matrix weightsWithoutRegularization;
	
	public String resultsDirectory;
	
	public LinearRegressor(LinearRegressorDataset lrDataset, int runNumber) {
		this.dataset = lrDataset;
		
		// Used every time getMSEGradient is calculated
		Matrix trainXTransposeDivideOneHalfN = lrDataset.trainX.transpose().timesEquals(2.0 / lrDataset.trainX.getRowDimension());
		trainXSquaredDivideOneHalfN = trainXTransposeDivideOneHalfN.times(lrDataset.trainX);
		trainXTransposeTimesTrainYDivideOneHalfN = trainXTransposeDivideOneHalfN.times(lrDataset.trainY);
		
		resultsDirectory = System.getProperty("user.dir") + "/results/";
	}
	
	// Error Calculations
	public double getMSE(Matrix X, Matrix Y, Matrix w) {
		return ExtraMatrixMethods.getSumOfSquares(X.times(w).minusEquals(Y)) / X.getRowDimension();
	}
	public double getRMSE(Matrix X, Matrix Y, Matrix w) {
		return Math.sqrt(getMSE(X, Y, w));
	}
	private Matrix getMSEGradient(Matrix w) {
		return trainXSquaredDivideOneHalfN.times(w).minusEquals(trainXTransposeTimesTrainYDivideOneHalfN);
	}
		
	public static String convertWeightsToTabSeparatedString(Matrix w) {
		StringBuffer retval = new StringBuffer();
		for (int i = 0; i < w.getRowDimension(); i++) {
			if (i != 0) {
				retval.append("\t");
			}
			retval.append(String.format("%f", w.get(i,0)));
		}
		return retval.toString();
	}
	
	public String runGradientDescentAndSaveResults(GradientDescentParameters parameters) {
		String directory = resultsDirectory + parameters.subDirectory;
		new File(directory).mkdirs();
		if (SimpleHostLock.checkDoneLock(directory + "doneLock.txt")) {
			return "Completed by another host";
		}
		if (!SimpleHostLock.checkAndClaimHostLock(directory + "hostLock.txt")) {
			return "Claimed by another host";
		}
		
		Matrix w = newRandomWeights(dataset.numberOfPredictorsPlus1);
		GradientDescentInformation info = new GradientDescentInformation(parameters, w);
		StopWatch globalTimer = new StopWatch().start();
		
		int i = 0;
		boolean breakEarly = false;
		for (; i < parameters.maxNumberOfIterations; i ++) {
			switch (parameters.updateRule) {
				case Original:
					w = updateWeightsWithRegularization(w, parameters.learningRate, parameters.lambda);
					break;
				case AdaptedLR:
					w = updateWeightsAdaptByMagnitudeOfGradientWithRegularization(w, parameters.learningRate, parameters.lambda);
					break;
			}
			
			info.trainingErrorByIteration[i] = getRMSE(dataset.trainX, dataset.trainY, w);
			info.validationErrorByIteration[i] = getRMSE(dataset.validX, dataset.validY, w);
			info.testErrorByIteration[i] = getRMSE(dataset.testX, dataset.testY, w);
			info.weightsByIteration[i] = w;
			
			if (DoubleCompare.lessThan(info.validationErrorByIteration[i], info.minValidationError)) {
				info.minValidationError = info.validationErrorByIteration[i];
				info.minValidationErrorIteration = i;
				info.iterationsSinceValidationErrorImproved[i] = 0;
			} else {
				info.iterationsSinceValidationErrorImproved[i] = info.iterationsSinceValidationErrorImproved[i-1]+1; 
			}
			if (DoubleCompare.lessThan(info.testErrorByIteration[i], info.minTestError)) {
				info.minTestError = info.testErrorByIteration[i];
				info.minTestErrorIteration = i;
			}
			if (DoubleCompare.lessThan(info.maxTrainingError, info.trainingErrorByIteration[i])) {
				info.maxTrainingError = info.trainingErrorByIteration[i];
				info.maxTrainingErrorIteration = i;
			}
			if (DoubleCompare.greaterThan(info.validationErrorByIteration[i], info.maxValidationError)) {
				info.maxValidationError = info.validationErrorByIteration[i];
				info.maxValidationErrorIteration = i;
			}
			if (DoubleCompare.greaterThan(info.testErrorByIteration[i], info.maxTestError)) {
				info.maxTestError = info.testErrorByIteration[i];
				info.maxTestErrorIteration = i;
			}
			
			if (DoubleCompare.lessThan(info.trainingErrorByIteration[i], info.minTrainingError)) {
				info.minTrainingError = info.trainingErrorByIteration[i];
				info.minTrainingErrorIteration = i;
				info.iterationsSinceTrainingErrorImproved[i] = 0;
			} else {
				info.iterationsSinceTrainingErrorImproved[i] = info.iterationsSinceTrainingErrorImproved[i-1]+1;  
				if (info.iterationsSinceTrainingErrorImproved[i] == info.maxIterationsSinceTrainingErrorImproved) {
					breakEarly = true;
				}
			}
			info.timeInSecondsUpToThisPoint[i] = globalTimer.getElapsedSeconds();
			if (breakEarly) {
				break;
			}
		}
		info.saveToFile(directory, i, dataset);
		if (SimpleHostLock.writeDoneLock(directory + "doneLock.txt")) {
			if (breakEarly) {
				return String.format("Gradient descent completed after reaching maxIterationsSinceTrainingErrorImproved of %d, stopping early after %d iterations.", 
						info.maxIterationsSinceTrainingErrorImproved, i);
			} else {
				return "Gradient descent completed after max iterations";
			}
		} else {
			return "Failed to write done lock for some reason.";
		}
	}
	
	private Matrix newRandomWeights(int numberOfWeights) {
		MersenneTwisterFast rand = new MersenneTwisterFast();
		double[][] weights = new double[numberOfWeights][1];
		for (int i = 0; i < numberOfWeights; i++) {
			weights[i][0] = rand.nextDouble();
		}
		return new Matrix(weights);
	}
	
	// When lambda is 0, these are the same as having no regularization so no need to distinguish
	private Matrix updateWeightsWithRegularization(Matrix w, double learningRate, double lambda) {
		return w.times(1 - ((2 * learningRate * lambda) / dataset.numberOfTrainingExamples)).minus(ExtraMatrixMethods.getUnitVector(getMSEGradient(w)).times(learningRate));
	}
	
	private Matrix updateWeightsAdaptByMagnitudeOfGradientWithRegularization(Matrix w, double learningRate, double lambda) {
		return w.times(1 - ((2 * learningRate * lambda) / dataset.numberOfTrainingExamples)).minus(getMSEGradient(w).times(learningRate));
	}

	private static class GradientDescentInformation {
		public GradientDescentParameters parameters;
		public Matrix initialWeights;
		public double[] trainingErrorByIteration;
		public double[] validationErrorByIteration;
		public double[] testErrorByIteration;
		public Matrix[] weightsByIteration;
		public double minTrainingError = Double.MAX_VALUE, minValidationError = Double.MAX_VALUE, minTestError = Double.MAX_VALUE,
				maxTrainingError = Double.MIN_VALUE, maxValidationError = Double.MIN_VALUE, maxTestError = Double.MIN_VALUE;
		public int minTrainingErrorIteration = 0, minValidationErrorIteration = 0, minTestErrorIteration = 0,
				   maxTrainingErrorIteration = 0, maxValidationErrorIteration = 0, maxTestErrorIteration = 0;
		public double[] timeInSecondsUpToThisPoint;
		
		// Use a ridiculously conservative stopping condition during the run. Once we have the data we can
		//	"simulate" what would have happened with various stopping conditions easily.
		public int maxIterationsSinceTrainingErrorImproved = 50000;
		public int[] iterationsSinceTrainingErrorImproved;
		
		// Wont be part of the current stopping condition since the goal is to just collect a bunch of data for
		//	later analysis. But will save us extra calculations then since this will certainly be part of the
		//	revised stopping condition.
		public int[] iterationsSinceValidationErrorImproved;
		
		private String fileName = "gradientDescent.txt";
		
		public GradientDescentInformation(GradientDescentParameters parameters, Matrix initialWeights) {
			this.parameters = parameters;
			this.initialWeights = initialWeights;
			
			trainingErrorByIteration = new double[parameters.maxNumberOfIterations];
			validationErrorByIteration = new double[parameters.maxNumberOfIterations];
			testErrorByIteration = new double[parameters.maxNumberOfIterations];
			weightsByIteration = new Matrix[parameters.maxNumberOfIterations];
			timeInSecondsUpToThisPoint = new double[parameters.maxNumberOfIterations];
			
			iterationsSinceTrainingErrorImproved = new int[parameters.maxNumberOfIterations];
			iterationsSinceValidationErrorImproved = new int[parameters.maxNumberOfIterations];
		}
		
		public void saveToFile(String directory, int actualNumberOfIterations, LinearRegressorDataset dataset) {			
			try {
				BufferedWriter bw = new BufferedWriter(new PrintWriter(directory + fileName));
				bw.write(parameters.prettyPrintOut);
				bw.write(String.format("ActualNumberOfIterations: %d\n", actualNumberOfIterations));
				bw.write(String.format("MaxIterationsSinceTrainingErrorImproved: %d\n", maxIterationsSinceTrainingErrorImproved));
				bw.write(String.format("NumberOfTrainingExamples: %d\n", dataset.numberOfAllTrainingExamples));
				bw.write(String.format("NumberOfValidationExamples: %d\n", dataset.numberOfValidationExamples));
				bw.write(String.format("NumberOfTestExamples: %d\n", dataset.numberOfTestExamples));
				bw.write(String.format("NumberOfWeights: %d\n", dataset.numberOfPredictorsPlus1));
				bw.write(String.format("IntialWeights: %s\n", convertWeightsToTabSeparatedString(initialWeights)));
				bw.write(String.format("TrainingRMSErrorMin: %d\t%f\n", minTrainingErrorIteration, minTrainingError));
				bw.write(String.format("TrainingRMSErrorMax: %d\t%f\n", maxTrainingErrorIteration, maxTrainingError));
				bw.write(String.format("ValidationRMSErrorMin: %d\t%f\n", minValidationErrorIteration, minValidationError));
				bw.write(String.format("ValidationRMSErrorMax: %d\t%f\n", maxValidationErrorIteration, maxValidationError));
				bw.write(String.format("TestRMSErrorMin: %d\t%f\n", minTestErrorIteration, minTestError));
				bw.write(String.format("TestRMSErrorMax: %d\t%f\n", maxTestErrorIteration, maxTestError));
				bw.write(String.format("IterationNumber\tTimeSoFarInSeconds\tTrainingError\tValidationError\tTestError\tIterationsSinceTrainingErrorImproved\tIterationsSinceValidationErrorImproved\tWeights\n"));
				for (int i = 0; i < actualNumberOfIterations; i++) {
					bw.write(String.format("%d\t%f\t%f\t%f\t%f\t%s\n", 
							i, timeInSecondsUpToThisPoint[i], trainingErrorByIteration[i], validationErrorByIteration[i], 
							testErrorByIteration[i], convertWeightsToTabSeparatedString(weightsByIteration[i])));
				}
				bw.flush();
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		

		
		public GradientDescentInformation(String directory) {
			// TODO
		}
	}
	
	/*
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
	*/
}
