package linearRegressor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import Jama.Matrix;
import utilities.DoubleCompare;
import utilities.ExtraMatrixMethods;
import utilities.StopWatch;
import utilities.SumCountAverage;

public class GradientDescentInformation {
	public GradientDescentSummary summary;
	
	public ArrayList<Double> trainingErrorByIteration;
	public ArrayList<Double> validationErrorByIteration;
	public ArrayList<Double> testErrorByIteration;
	public ArrayList<Matrix> weightsByIteration;

	public ArrayList<Double> timeInSecondsUpToThisPoint;
	
	public ArrayList<Integer> iterationsSinceTrainingErrorImproved;
	
	// Wont be part of the current stopping condition since the goal is to just collect a bunch of data for
	//	later analysis. But will save us extra calculations then since this will certainly be part of the
	//	revised stopping condition.
	public ArrayList<Integer> iterationsSinceValidationErrorImproved;
	
	private String fileName = "gradientDescent.txt";
	
	public GradientDescentInformation(GradientDescentParameters parameters) {
		this.summary = new GradientDescentSummary(parameters);
		
		trainingErrorByIteration = new ArrayList<Double>();
		validationErrorByIteration = new ArrayList<Double>();
		testErrorByIteration = new ArrayList<Double>();
		weightsByIteration = new ArrayList<Matrix>();
		timeInSecondsUpToThisPoint = new ArrayList<Double>();
		
		iterationsSinceTrainingErrorImproved = new ArrayList<Integer>();
		iterationsSinceValidationErrorImproved = new ArrayList<Integer>();
	}
	
	public void printStatusMessage(String message, StopWatch timer) {
		
		timer.printMessageWithTime(summary.parameters.getMinimalDescription() + "\n\t" + 
				message + "\n\t" +
				String.format("MinTrainingError: %f MinValidationError: %f", summary.minTrainingError, summary.minValidationError));
	}
	
	public void addTrainingError(double error) {
		trainingErrorByIteration.add(error);
		int iter = trainingErrorByIteration.size() - 1;
		if (DoubleCompare.lessThan(summary.maxTrainingError, error)) {
			summary.maxTrainingError = error;
			summary.maxTrainingErrorIteration = iter;
		}
		
		if (DoubleCompare.lessThan(error, summary.minTrainingError)) {
			summary.minTrainingError = error;
			summary.minTrainingErrorIteration = iter;
			iterationsSinceTrainingErrorImproved.add(0);
		} else {
			int newIterSinceImpr = iterationsSinceTrainingErrorImproved.get(iter-1)+1;
			iterationsSinceTrainingErrorImproved.add(newIterSinceImpr);
		}
	}
	
	public void addValidationError(double error) {
		validationErrorByIteration.add(error);
		int iter = validationErrorByIteration.size() - 1;
		if (DoubleCompare.lessThan(summary.maxValidationError, error)) {
			summary.maxValidationError = error;
			summary.maxValidationErrorIteration = iter;
		}
		
		if (DoubleCompare.lessThan(error, summary.minValidationError)) {
			summary.minValidationError = error;
			summary.minValidationErrorIteration = iter;
			iterationsSinceValidationErrorImproved.add(0);
		} else {
			int newIterSinceImpr = iterationsSinceValidationErrorImproved.get(iter-1)+1;
			iterationsSinceValidationErrorImproved.add(newIterSinceImpr);
		}
	}
	
	public void addTestError(double error) {
		testErrorByIteration.add(error);
		int iter = testErrorByIteration.size() - 1;
		if (DoubleCompare.lessThan(summary.maxTestError, error)) {
			summary.maxTestError = error;
			summary.maxTestErrorIteration = iter;
		}
		
		if (DoubleCompare.lessThan(error, summary.minTestError)) {
			summary.minTestError = error;
			summary.minTestErrorIteration = iter;
		}
	}
	
	public boolean isTimeToStopBasedOnTrainingError() {
		int lastIteration =  trainingErrorByIteration.size() - 1;
		if (iterationsSinceTrainingErrorImproved.get(lastIteration) > GradientDescentParameters.maxIterationsSinceTrainingErrorImproved) {
			System.out.println(String.format("Gradient descent completed after reaching maxIterationsSinceTrainingErrorImproved of %d, stopping early after %d iterations.", 
					GradientDescentParameters.maxIterationsSinceTrainingErrorImproved, trainingErrorByIteration.size()));
			return true;
		}
		return false;
	}
	
	public void saveToFile() {	
		summary.saveToFile();
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(summary.directory + fileName));
			bw.write(String.format("IterationNumber\t"
					+ "TimeSoFarInSeconds\t"
					+ "TrainingError\t"
					+ "ValidationError\t"
					+ "TestError\t"
					+ "Weights\n"));
			for (int i = 0; i < summary.actualNumberOfIterations; i++) {
				bw.write(String.format("%d\t%f\t%f\t%f\t%f\t%s\n", 
						i, 
						timeInSecondsUpToThisPoint.get(i), 
						trainingErrorByIteration.get(i), 
						validationErrorByIteration.get(i),
						testErrorByIteration.get(i), 
						ExtraMatrixMethods.convertWeightsToTabSeparatedString(weightsByIteration.get(i))));
			}
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void generateErrorCurve() {	
		// TOOD

	}
	
	public static GradientDescentInformation readFromFile(GradientDescentParameters parameters, boolean averages) {	
		GradientDescentInformation retval = new GradientDescentInformation(parameters);
		retval.summary = GradientDescentSummary.readFromFile(parameters, averages);
		try {
			BufferedReader br = new BufferedReader(new FileReader(retval.summary.directory + retval.fileName));
			br.readLine(); // Skip the header 
			for (int i = 0; i < retval.summary.actualNumberOfIterations; i++) {
				String[] components = br.readLine().split("\t");
				retval.timeInSecondsUpToThisPoint.add(Double.parseDouble(components[1]));
				retval.addTrainingError(Double.parseDouble(components[2]));
				retval.addValidationError(Double.parseDouble(components[3]));
				retval.addTestError(Double.parseDouble(components[4]));
				
				double[] weights = new double[parameters.dataset.numberOfPredictorsPlus1];
				String[] weightsLine = br.readLine().split(": ");
				for (int j = 0; j < weights.length; j++) {
					weights[j] = Double.parseDouble(weightsLine[j+1]);
				}
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return retval;
	}
	
	public GradientDescentInformation averageInformation(GradientDescentInformation... infos) {
		GradientDescentInformation retval = new GradientDescentInformation(infos[0].summary.parameters);
		GradientDescentSummary[] summaries = new GradientDescentSummary[infos.length];
		for (int i = 0; i < infos.length; i++) {
			summaries[i] = infos[i].summary;
		}
		retval.summary = GradientDescentSummary.averageSummary(summaries);
		SumCountAverage avgTrainingError = new SumCountAverage(), avgValidationError = new SumCountAverage(), avgTestError = new SumCountAverage();
		Matrix avgWeights = null;
		for (GradientDescentInformation info : infos) {
			for (int i =0; i < retval.summary.minimumIterationsAllRunsShare; i++) {
				avgTrainingError.reset();
				avgValidationError.reset();
				avgTestError.reset();
				avgWeights = new Matrix(retval.summary.initialWeights.getRowDimension(),retval.summary.initialWeights.getColumnDimension());
				for (int runNum = 0; runNum < infos.length; runNum++) {
					avgTrainingError.addData(info.trainingErrorByIteration.get(i));
					avgValidationError.addData(info.validationErrorByIteration.get(i));
					avgTestError.addData(info.testErrorByIteration.get(i));
					avgWeights.plusEquals(info.weightsByIteration.get(i));
				}
				retval.addTrainingError(avgTrainingError.getMean());
				retval.addValidationError(avgValidationError.getMean());
				retval.addTestError(avgTestError.getMean());
				retval.weightsByIteration.add(avgWeights.timesEquals(1.0 / infos.length));
			}
		}

		return retval;
	}
}
