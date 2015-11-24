package linearRegressor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

import Jama.Matrix;
import utilities.ExtraMatrixMethods;

public class GradientDescentSummary {
	public GradientDescentParameters parameters;
	public String directory; 
	
	public GradientDescentSummary(GradientDescentParameters parameters) {
		this(parameters, false);
	}
	
	public GradientDescentSummary(GradientDescentParameters parameters, boolean averages) {
		this.parameters = parameters;
		this.directory = Main.RESULTS_DIRECTORY + ((averages) ? "Averages/" : "") + parameters.subDirectory;
	}
	
	public Matrix initialWeights;
	public double minTrainingError = Double.MAX_VALUE, minValidationError = Double.MAX_VALUE, minTestError = Double.MAX_VALUE,
			maxTrainingError = Double.MIN_VALUE, maxValidationError = Double.MIN_VALUE, maxTestError = Double.MIN_VALUE,
			minTrainingErrorIteration = 0, minValidationErrorIteration = 0, minTestErrorIteration = 0,
			maxTrainingErrorIteration = 0, maxValidationErrorIteration = 0, maxTestErrorIteration = 0,
			actualNumberOfIterations = 0, minimumIterationsAllRunsShare = 0;
	
	private String fileName = "summary.txt";
	
	public static GradientDescentSummary averageSummary(GradientDescentSummary... summaries) {
		
		GradientDescentSummary retval = new GradientDescentSummary(summaries[0].parameters, true);
		retval.initialWeights =  new Matrix(summaries[0].initialWeights.getRowDimension(), summaries[0].initialWeights.getColumnDimension());
		for (GradientDescentSummary summary : summaries) {
			retval.minTrainingError += summary.minTrainingError;
			retval.minTrainingErrorIteration += summary.minTrainingErrorIteration;
			retval.minValidationError += summary.minValidationError;
			retval.minValidationErrorIteration += summary.minTestErrorIteration;
			retval.minTestError += summary.minValidationError;
			retval.minTestErrorIteration += summary.minTestErrorIteration;
			retval.maxTrainingError += summary.maxTrainingError;
			retval.maxTrainingErrorIteration += summary.maxTrainingErrorIteration;
			retval.maxValidationError += summary.maxValidationError;
			retval.maxValidationErrorIteration += summary.minTestErrorIteration;
			retval.maxTestError += summary.maxValidationError;
			retval.maxTestErrorIteration += summary.maxTestErrorIteration;
			retval.actualNumberOfIterations += summary.actualNumberOfIterations;
			if (summary.actualNumberOfIterations < retval.minimumIterationsAllRunsShare) {
				retval.minimumIterationsAllRunsShare = summary.actualNumberOfIterations;
			}
			retval.initialWeights.plusEquals(summary.initialWeights);
		}
		retval.minTrainingError /= summaries.length;
		retval.minTrainingErrorIteration /= summaries.length;
		retval.minValidationError /= summaries.length;
		retval.minValidationErrorIteration /= summaries.length;
		retval.minTestError /= summaries.length;
		retval.minTestErrorIteration /= summaries.length;
		retval.maxTrainingError /= summaries.length;
		retval.maxTrainingErrorIteration /= summaries.length;
		retval.maxValidationError /= summaries.length;
		retval.maxValidationErrorIteration /= summaries.length;
		retval.maxTestError /= summaries.length;
		retval.maxTestErrorIteration /= summaries.length;
		retval.actualNumberOfIterations /= summaries.length;
		retval.initialWeights = retval.initialWeights.timesEquals(1.0 / summaries.length);
		return retval;
	}
	
	public void saveToFile() {			
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(directory + fileName));
			bw.write(parameters.prettyPrintOut);
			bw.write(String.format("ActualNumberOfIterations: %.2f\n", actualNumberOfIterations));
			bw.write(String.format("TrainingRMSEMin: %.2f\t%f\n", minTrainingErrorIteration, minTrainingError));
			bw.write(String.format("TrainingRMSEMax: %.2f\t%f\n", maxTrainingErrorIteration, maxTrainingError));
			bw.write(String.format("ValidationRMSEMin: %.2f\t%f\n", minValidationErrorIteration, minValidationError));
			bw.write(String.format("ValidationRMSEMax: %.2f\t%f\n", maxValidationErrorIteration, maxValidationError));
			bw.write(String.format("TestRMSEMin: %.2f\t%f\n", minTestErrorIteration, minTestError));
			bw.write(String.format("TestRMSEMax: %.2f\t%f\n", maxTestErrorIteration, maxTestError));
			bw.write(String.format("IntialWeights: %s\n", ExtraMatrixMethods.convertWeightsToTabSeparatedString(initialWeights)));

			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static GradientDescentSummary readFromFile(GradientDescentParameters parameters, boolean averages) {			
		GradientDescentSummary retval = new GradientDescentSummary(parameters, averages);
		try {
			BufferedReader br = new BufferedReader(new FileReader(retval.directory + retval.fileName));
			String line;
			while(!(line = br.readLine()).startsWith("ActualNumberOfIterations"));
			
			retval.actualNumberOfIterations = Double.parseDouble(line.split(": ")[1]);
			
			String[] components = br.readLine().split(": ");
			retval.minTrainingErrorIteration = Double.parseDouble(components[1]);
			retval.minTrainingError = Double.parseDouble(components[2]);
			components = br.readLine().split(": ");
			retval.maxTrainingErrorIteration = Double.parseDouble(components[1]);
			retval.maxTrainingError = Double.parseDouble(components[2]);
			components = br.readLine().split(": ");
			retval.minValidationErrorIteration = Double.parseDouble(components[1]);
			retval.minValidationError = Double.parseDouble(components[2]);
			components = br.readLine().split(": ");
			retval.maxValidationErrorIteration = Double.parseDouble(components[1]);
			retval.maxValidationError = Double.parseDouble(components[2]);
			components = br.readLine().split(": ");
			retval.minTestErrorIteration = Double.parseDouble(components[1]);
			retval.minTestError = Double.parseDouble(components[2]);
			components = br.readLine().split(": ");
			retval.maxTestErrorIteration = Double.parseDouble(components[1]);
			retval.maxTestError = Double.parseDouble(components[2]);
			double[] initialWeights = new double[parameters.dataset.numberOfPredictorsPlus1];
			String[] initialWeightsLine = br.readLine().split(": ");
			for (int i = 0; i < initialWeights.length; i++) {
				initialWeights[i] = Double.parseDouble(initialWeightsLine[i+1]);
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return retval;
	}
}
