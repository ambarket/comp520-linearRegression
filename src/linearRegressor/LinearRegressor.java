package linearRegressor;
import utilities.ExtraMatrixMethods;
import utilities.MersenneTwisterFast;
import utilities.StopWatch;
import Jama.Matrix;
import dataset.LinearRegressorDataset;
enum UpdateRule {Original, AdaptedLR}
public class LinearRegressor {
	
	public LinearRegressorDataset dataset;
	public Matrix trainXSquaredDivideOneHalfN;
	public Matrix trainXTransposeTimesTrainYDivideOneHalfN;
	
	public Matrix weightsWithoutRegularization;
	
	public LinearRegressor(LinearRegressorDataset lrDataset) {
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
	
	public GradientDescentInformation runGradientDescent(GradientDescentParameters parameters) {

		
		Matrix w = newRandomWeights(dataset.numberOfPredictorsPlus1);
		GradientDescentInformation info = new GradientDescentInformation(parameters);
		info.summary.initialWeights = w;
		StopWatch globalTimer = new StopWatch().start();

		int i = 0;
		for (; i < parameters.maxNumberOfIterations; i ++) {
			Matrix gradient = getMSEGradient(w);

			switch (parameters.updateRule) {
				case Original:
					w = updateWeightsWithRegularization(w, parameters.learningRate, parameters.lambda, gradient);
					break;
				case AdaptedLR:
					w = updateWeightsAdaptByMagnitudeOfGradientWithRegularization(w, parameters.learningRate, parameters.lambda, gradient, i);
					break;
			}
			
			
			info.addTrainingError(getRMSE(dataset.trainX, dataset.trainY, w));
			info.addValidationError(getRMSE(dataset.validX, dataset.validY, w));
			info.addTestError(getRMSE(dataset.testX, dataset.testY, w));
			info.addGradientMagnitude(ExtraMatrixMethods.getL2Norm(gradient));
			info.weightsByIteration.add(w);
			info.timeInSecondsUpToThisPoint.add(globalTimer.getElapsedSeconds());
			
			if (i % 20000 == 0) {
				info.printStatusMessage("Completed " + i + " iterations", globalTimer);
			}
			if (info.allStoppingConditionsHaveBeenMet()) {
				break;
			}
		}
		info.summary.actualNumberOfIterations = i;
		return info;
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
	private Matrix updateWeightsWithRegularization(Matrix w, double learningRate, double lambda, Matrix gradient) {
		gradient = ExtraMatrixMethods.getUnitVector(gradient);
		return w.times(1 - ((2 * learningRate * lambda) / dataset.numberOfTrainingExamples)).minus(gradient.times(learningRate));
	}
	
	private Matrix updateWeightsAdaptByMagnitudeOfGradientWithRegularization(Matrix w, double learningRate, double lambda, Matrix gradient, int iteration) {
		gradient = ExtraMatrixMethods.getUnitVector(getMSEGradient(w));
		learningRate = learningRate / (iteration / 10000 + 1);
		return w.times(1 - ((2 * learningRate * lambda) / dataset.numberOfTrainingExamples)).minus(gradient.times(learningRate));
	}
}
