package linearRegressor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Comparator;
import java.util.TreeSet;

import Jama.Matrix;
import dataset.Dataset;
import dataset.DatasetParameters;
import dataset.LinearRegressorDataset;
import utilities.StopWatch;

public class GradientDescentSummary implements Comparable<GradientDescentSummary> {
	public GradientDescentParameters parameters;
	public String directory; 
	
	public GradientDescentSummary(GradientDescentParameters parameters) {
		this.parameters = parameters;
		this.directory = Main.RESULTS_DIRECTORY + parameters.subDirectory;
		new File(this.directory).mkdirs();
	}
	
	public Matrix initialWeights;
	public double minTrainingError = Double.MAX_VALUE, minValidationError = Double.MAX_VALUE, minTestError = Double.MAX_VALUE,
			maxTrainingError = Double.MIN_VALUE, maxValidationError = Double.MIN_VALUE, maxTestError = Double.MIN_VALUE,
			minTrainingErrorIteration = 0, minValidationErrorIteration = 0, minTestErrorIteration = 0,
			maxTrainingErrorIteration = 0, maxValidationErrorIteration = 0, maxTestErrorIteration = 0,
			minGradientMagnitude = Double.MAX_VALUE, minGradientMagnitudeIteration = 0,
			maxGradientMagnitude = Double.MIN_VALUE, maxGradientMagnitudeIteration = 0,
			actualNumberOfIterations = 0, minimumIterationsAllRunsShare = Double.MAX_VALUE;
	
	public double trainingStoppingIteration = -1, validationStoppingIteration = -1, gradientStoppingIteration = -1,
			trainingStoppingTrainingError = -1, validationStoppingTrainingError  = -1, gradientStoppingTrainingError  = -1,
			trainingStoppingValidationError = -1, validationStoppingValidationError = -1, gradientStoppingValidationError = -1,
			trainingStoppingTestError = -1, validationStoppingTestError = -1, gradientStoppingTestError = -1;
	
	public double timeInSeconds = 0;
	
	private String fileName = "summary.txt";
	
	public static GradientDescentSummary averageSummary(GradientDescentParameters parameters, GradientDescentSummary... summaries) {
		
		GradientDescentSummary retval = new GradientDescentSummary(parameters);
		//retval.initialWeights =  new Matrix(summaries[0].initialWeights.getRowDimension(), summaries[0].initialWeights.getColumnDimension());
		for (GradientDescentSummary summary : summaries) {
			/*
			retval.minTrainingError += summary.minTrainingError;
			retval.minTrainingErrorIteration += summary.minTrainingErrorIteration;
			
			retval.minValidationError += summary.minValidationError;
			retval.minValidationErrorIteration += summary.minValidationErrorIteration;
			
			retval.minTestError += summary.minTestError;
			retval.minTestErrorIteration += summary.minTestErrorIteration;
			
			retval.maxTrainingError += summary.maxTrainingError;
			retval.maxTrainingErrorIteration += summary.maxTrainingErrorIteration;
			
			retval.maxValidationError += summary.maxValidationError;
			retval.maxValidationErrorIteration += summary.maxValidationErrorIteration;
			
			retval.maxTestError += summary.maxTestError;
			retval.maxTestErrorIteration += summary.maxTestErrorIteration;
			
			retval.actualNumberOfIterations += summary.actualNumberOfIterations;
			
			retval.minGradientMagnitude += summary.minGradientMagnitude;
			retval.minGradientMagnitudeIteration += summary.minGradientMagnitudeIteration;
			
			retval.maxGradientMagnitude += summary.maxGradientMagnitude;
			retval.maxGradientMagnitudeIteration += summary.maxGradientMagnitudeIteration;
			
			retval.trainingStoppingIteration += summary.trainingStoppingIteration;
			retval.validationStoppingIteration += summary.validationStoppingIteration;
			retval.gradientStoppingIteration += summary.gradientStoppingIteration;
			
			retval.trainingStoppingTrainingError += summary.trainingStoppingTrainingError;
			retval.validationStoppingTrainingError += summary.validationStoppingTrainingError;
			retval.gradientStoppingTrainingError += summary.gradientStoppingTrainingError;
			
			retval.trainingStoppingValidationError += summary.trainingStoppingValidationError;
			retval.validationStoppingValidationError += summary.validationStoppingValidationError;
			retval.gradientStoppingValidationError += summary.gradientStoppingValidationError;
			
			retval.trainingStoppingTestError += summary.trainingStoppingTestError;
			retval.validationStoppingTestError += summary.validationStoppingTestError;
			retval.gradientStoppingTestError += summary.gradientStoppingTestError;
			*/
			if (summary.actualNumberOfIterations < retval.minimumIterationsAllRunsShare) {
				retval.minimumIterationsAllRunsShare = summary.actualNumberOfIterations;
			}
			
			retval.timeInSeconds += summary.timeInSeconds;
			//retval.initialWeights.plusEquals(summary.initialWeights);
		}
		/*
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
		
		retval.minGradientMagnitude /= summaries.length;
		retval.minGradientMagnitudeIteration /= summaries.length;
		
		retval.maxGradientMagnitude /= summaries.length;
		retval.maxGradientMagnitudeIteration /= summaries.length;
		
		retval.trainingStoppingIteration /= summaries.length;
		retval.validationStoppingIteration /= summaries.length;
		retval.gradientStoppingIteration /= summaries.length;
		
		retval.trainingStoppingTrainingError /= summaries.length;
		retval.validationStoppingTrainingError /= summaries.length;
		retval.gradientStoppingTrainingError /= summaries.length;
		
		retval.trainingStoppingValidationError /= summaries.length;
		retval.validationStoppingValidationError /= summaries.length;
		retval.gradientStoppingValidationError /= summaries.length;
		
		retval.trainingStoppingTestError /= summaries.length;
		retval.validationStoppingTestError /= summaries.length;
		retval.gradientStoppingTestError /= summaries.length;
		*/
		retval.actualNumberOfIterations = retval.minimumIterationsAllRunsShare;
		retval.timeInSeconds /= summaries.length;
		//retval.initialWeights = retval.initialWeights.timesEquals(1.0 / summaries.length);
		return retval;
	}
	
	public void saveToFile() {			
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(directory + fileName));
			bw.write(parameters.prettyPrintOut);
			bw.write(String.format("TimeInSeconds: %.2f\n", timeInSeconds));
			bw.write(String.format("ActualNumberOfIterations: %.2f\n", actualNumberOfIterations));
			bw.write(String.format("TrainingRMSEMin: %.2f\t%f\n", minTrainingErrorIteration, minTrainingError));
			bw.write(String.format("TrainingRMSEMax: %.2f\t%f\n", maxTrainingErrorIteration, maxTrainingError));
			bw.write(String.format("ValidationRMSEMin: %.2f\t%f\n", minValidationErrorIteration, minValidationError));
			bw.write(String.format("ValidationRMSEMax: %.2f\t%f\n", maxValidationErrorIteration, maxValidationError));
			bw.write(String.format("TestRMSEMin: %.2f\t%f\n", minTestErrorIteration, minTestError));
			bw.write(String.format("TestRMSEMax: %.2f\t%f\n", maxTestErrorIteration, maxTestError));
			bw.write(String.format("GradientMagnitudeMin: %.2f\t%f\n", minGradientMagnitudeIteration, minGradientMagnitude));
			bw.write(String.format("GradientMagnitudeMax: %.2f\t%f\n", maxGradientMagnitudeIteration, maxGradientMagnitude));
			bw.write(String.format("TrainingStopping(iter train valid test): %.2f\t%f\t%f\t%f\n", trainingStoppingIteration, trainingStoppingTrainingError, trainingStoppingValidationError, trainingStoppingTestError));
			bw.write(String.format("ValidationStopping(iter train valid test): %.2f\t%f\t%f\t%f\n", validationStoppingIteration, validationStoppingTrainingError, validationStoppingValidationError, validationStoppingTestError));
			bw.write(String.format("GradientStopping(iter train valid test): %.2f\t%f\t%f\t%f\n", gradientStoppingIteration, gradientStoppingTrainingError, gradientStoppingValidationError, gradientStoppingTestError));

			//bw.write(String.format("IntialWeights: %s\n", ExtraMatrixMethods.convertWeightsToTabSeparatedString(initialWeights)));

			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void writeSortedSummaryRecordHeader(BufferedWriter bw) throws IOException {
		bw.write(String.format("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", 
				GradientDescentParameters.getTabSeparatedMinimalPrintoutHeader(),
				"minValidationError", 
				"minTestError", 
				"minTrainingError", 
				"minValidationErrorIteration", 
				"minTestErrorIteration", 
				"minTrainingErrorIteration", 
				"validationStoppingIteration", 
				"gradientStoppingIteration", 
				"trainingStoppingIteration"
				));
	}
	public void writeSortedSummaryRecord(BufferedWriter bw) throws IOException {	
		bw.write(String.format("%s\t%f\t%f\t%f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\n", 
				parameters.getTabSeparatedMinimalPrintout(),
				minValidationError, 
				minTestError, 
				minTrainingError, 
				minValidationErrorIteration, 
				minTestErrorIteration, 
				minTrainingErrorIteration, 
				validationStoppingIteration, 
				gradientStoppingIteration, 
				trainingStoppingIteration
				));
	}
	
	public static GradientDescentSummary readFromFile(GradientDescentParameters parameters) {			
		GradientDescentSummary retval = new GradientDescentSummary(parameters);
		try {
			BufferedReader br = new BufferedReader(new FileReader(retval.directory + retval.fileName));
			String line;
			while(!(line = br.readLine()).startsWith("ActualNumberOfIterations") && !(line).startsWith("TimeInSeconds"));
			
			if (line.startsWith("TimeInSeconds")) {
				retval.timeInSeconds = Double.parseDouble(line.split(": ")[1]);
				line = br.readLine();
			}
			retval.actualNumberOfIterations = Double.parseDouble(line.split(": ")[1]);
			
			String[] components = br.readLine().split(": ")[1].split("\t");
			retval.minTrainingErrorIteration = Double.parseDouble(components[0]);
			retval.minTrainingError = Double.parseDouble(components[1]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.maxTrainingErrorIteration = Double.parseDouble(components[0]);
			retval.maxTrainingError = Double.parseDouble(components[1]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.minValidationErrorIteration = Double.parseDouble(components[0]);
			retval.minValidationError = Double.parseDouble(components[1]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.maxValidationErrorIteration = Double.parseDouble(components[0]);
			retval.maxValidationError = Double.parseDouble(components[1]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.minTestErrorIteration = Double.parseDouble(components[0]);
			retval.minTestError = Double.parseDouble(components[1]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.maxTestErrorIteration = Double.parseDouble(components[0]);
			retval.maxTestError = Double.parseDouble(components[1]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.minGradientMagnitudeIteration = Double.parseDouble(components[0]);
			retval.minGradientMagnitude = Double.parseDouble(components[1]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.maxGradientMagnitudeIteration = Double.parseDouble(components[0]);
			retval.maxGradientMagnitude = Double.parseDouble(components[1]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.trainingStoppingIteration = Double.parseDouble(components[0]);
			retval.trainingStoppingTrainingError = Double.parseDouble(components[1]);
			retval.trainingStoppingValidationError = Double.parseDouble(components[2]);
			retval.trainingStoppingTestError = Double.parseDouble(components[3]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.validationStoppingIteration = Double.parseDouble(components[0]);
			retval.validationStoppingTrainingError = Double.parseDouble(components[1]);
			retval.validationStoppingValidationError = Double.parseDouble(components[2]);
			retval.validationStoppingTestError = Double.parseDouble(components[3]);
			components = br.readLine().split(": ")[1].split("\t");
			retval.gradientStoppingIteration = Double.parseDouble(components[0]);
			retval.gradientStoppingTrainingError = Double.parseDouble(components[1]);
			retval.gradientStoppingValidationError = Double.parseDouble(components[2]);
			retval.gradientStoppingTestError = Double.parseDouble(components[3]);
			//double[] initialWeights = new double[parameters.dataset.numberOfPredictorsPlus1];
			//String[] initialWeightsLine = br.readLine().split(": ")[1].split("\t");
			//for (int i = 0; i < initialWeights.length; i++) {
			//	initialWeights[i] = Double.parseDouble(initialWeightsLine[i]);
			//}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return retval;
	}
	
	public static TreeSet<GradientDescentSummary> readGradientDescentSummaries(DatasetParameters dsParam) {
		LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, Main.TRAINING_SAMPLE_FRACTION));
		int tmp = (int)(unpartitionedDataset.numberOfAllTrainingExamples * (1 - Main.VALIDATION_SAMPLE_FRACTION));
		LinearRegressor lr = new LinearRegressor( new LinearRegressorDataset(unpartitionedDataset, tmp));
		TreeSet<GradientDescentSummary> sorted = new TreeSet<>();
		
		String resultsSubDirectory = String.format("%s/gradientDescent/Averages/%s/", 
				dsParam.minimalName,
				"StandardValidation"
			);
		
		for (UpdateRule updateRule : Main.updateRules) {
			for (double learningRate : Main.learningRates) {
				for (double lambda : Main.lambdas) {
					sorted.add(GradientDescentSummary.readFromFile(new GradientDescentParameters(resultsSubDirectory,
											lr.dataset, 
											Main.maxNumberOfIterations, 
											updateRule, 
											learningRate, 
											lambda)));
				}
			}
		}
		return sorted;
	}
	public static void readSortAndSaveGradientDescentSummaries() {
		
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : Main.datasets) {

			TreeSet<GradientDescentSummary> sorted = readGradientDescentSummaries(dsParam);
		
			String subdirectory = String.format("%s/gradientDescent/Averages/%s/",
					dsParam.minimalName,
					"StandardValidation"
				);
			try {
				BufferedWriter bw = new BufferedWriter(new PrintWriter(Main.RESULTS_DIRECTORY + subdirectory + "sortedGradientDescentSummaries.txt"));
				GradientDescentSummary.writeSortedSummaryRecordHeader(bw);
				for (GradientDescentSummary summary : sorted) {
					summary.writeSortedSummaryRecord(bw);
				}
				bw.flush();
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(1);
			}

			runTimer.printMessageWithTime("Finished run " + 0);
		}
	}

	@Override
	public int compareTo(GradientDescentSummary arg0) {
		return Double.compare(this.minValidationError, arg0.minValidationError);
	}
	
	public static class ValidationStopComparator implements Comparator<GradientDescentSummary> {
		@Override
		public int compare(GradientDescentSummary o1, GradientDescentSummary o2) {
			return Double.compare(o1.validationStoppingValidationError, o2.validationStoppingValidationError);
		}
	}
	
	public static class GradientStopComparator implements Comparator<GradientDescentSummary> {
		@Override
		public int compare(GradientDescentSummary o1, GradientDescentSummary o2) {
			return Double.compare(o1.gradientStoppingValidationError, o2.gradientStoppingValidationError);
		}
	}
}
