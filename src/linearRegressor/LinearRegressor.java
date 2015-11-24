package linearRegressor;
import java.io.File;

import Jama.Matrix;
import dataset.LinearRegressorDataset;
import utilities.ExtraMatrixMethods;
import utilities.MersenneTwisterFast;
import utilities.SimpleHostLock;
import utilities.StopWatch;
enum UpdateRule {Original, AdaptedLR}
public class LinearRegressor {
	
	public LinearRegressorDataset dataset;
	public Matrix trainXSquaredDivideOneHalfN;
	public Matrix trainXTransposeTimesTrainYDivideOneHalfN;
	
	public Matrix weightsWithoutRegularization;
	
	public LinearRegressor(LinearRegressorDataset lrDataset, int runNumber) {
		this.dataset = lrDataset;
		
		// Used every time getMSEGradient is calculated
		Matrix trainXTransposeDivideOneHalfN = lrDataset.trainX.transpose().timesEquals(2.0 / lrDataset.trainX.getRowDimension());
		trainXSquaredDivideOneHalfN = trainXTransposeDivideOneHalfN.times(lrDataset.trainX);
		trainXTransposeTimesTrainYDivideOneHalfN = trainXTransposeDivideOneHalfN.times(lrDataset.trainY);
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
	
	public String runGradientDescentAndSaveResults(GradientDescentParameters parameters) {
		String directory = Main.RESULTS_DIRECTORY + parameters.subDirectory;
		new File(directory).mkdirs();
		if (SimpleHostLock.checkDoneLock(directory + "doneLock.txt")) {
			return "Completed by another host";
		}
		if (!SimpleHostLock.checkAndClaimHostLock(directory + "hostLock.txt")) {
			return "Claimed by another host";
		}
		
		Matrix w = newRandomWeights(dataset.numberOfPredictorsPlus1);
		GradientDescentInformation info = new GradientDescentInformation(parameters);
		info.summary.initialWeights = w;
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
			
			
			info.addTrainingError(getRMSE(dataset.trainX, dataset.trainY, w));
			info.addValidationError(getRMSE(dataset.validX, dataset.validY, w));
			info.addTestError(getRMSE(dataset.testX, dataset.testY, w));
			info.weightsByIteration.add(w);
			info.timeInSecondsUpToThisPoint.add(globalTimer.getElapsedSeconds());
			
			if (i % 20000 == 0) {
				info.printStatusMessage("Completed " + i + " iterations", globalTimer);
			}
			if (info.isTimeToStopBasedOnTrainingError()) {
				break;
			}
		}
		info.summary.actualNumberOfIterations = i;
		info.saveToFile();
		if (SimpleHostLock.writeDoneLock(directory + "doneLock.txt")) {
			return "Gradient descent completed";
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
}
