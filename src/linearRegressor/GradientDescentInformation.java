package linearRegressor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import utilities.CommandLineExecutor;
import utilities.DoubleCompare;
import utilities.ExtraMatrixMethods;
import utilities.MathematicaListCreator;
import utilities.StopWatch;
import utilities.SumCountAverage;
import Jama.Matrix;

public class GradientDescentInformation {
	public GradientDescentSummary summary;
	
	public ArrayList<Double> trainingErrorByIteration;
	public ArrayList<Double> validationErrorByIteration;
	public ArrayList<Double> testErrorByIteration;
	public ArrayList<Matrix> weightsByIteration;

	public ArrayList<Double> timeInSecondsUpToThisPoint;
	
	public ArrayList<Integer> iterationsSinceTrainingErrorImproved;
	
	public ArrayList<Double> gradientMagnitudesByIteration;
	public ArrayList<Integer> iterationsSinceGradientMagnitudeDecreased;
	
	// Wont be part of the current stopping condition since the goal is to just collect a bunch of data for
	//	later analysis. But will save us extra calculations then since this will certainly be part of the
	//	revised stopping condition.
	public ArrayList<Integer> iterationsSinceValidationErrorImproved;
	
	private String fileName = "gradientDescent.txt", errorCurveScriptFileName = "errorCurveScript.m", latexCodeFileName = "latexCode.txt";
	
	public GradientDescentInformation(GradientDescentParameters parameters) {
		this.summary = new GradientDescentSummary(parameters);
		
		trainingErrorByIteration = new ArrayList<Double>();
		validationErrorByIteration = new ArrayList<Double>();
		testErrorByIteration = new ArrayList<Double>();
		
		gradientMagnitudesByIteration = new ArrayList<Double>();
		
		weightsByIteration = new ArrayList<Matrix>();
		timeInSecondsUpToThisPoint = new ArrayList<Double>();
		
		iterationsSinceTrainingErrorImproved = new ArrayList<Integer>();
		iterationsSinceValidationErrorImproved = new ArrayList<Integer>();
		iterationsSinceGradientMagnitudeDecreased = new ArrayList<Integer>();
	}
	
	public void printStatusMessage(String message, StopWatch timer) {
		
		timer.printMessageWithTime(summary.parameters.getMinimalDescription() + "\n\t" + 
				message + "\n\t" +
				String.format("MinTrainingError: %f MinValidationError: %f", summary.minTrainingError, summary.minValidationError));
	}
	
	public void addGradientMagnitude(double mag) {
		gradientMagnitudesByIteration.add(mag);
		int iter = trainingErrorByIteration.size() - 1;
		
		if (DoubleCompare.lessThan(summary.maxGradientMagnitude, mag)) {
			summary.maxGradientMagnitude = mag;
			summary.maxGradientMagnitudeIteration = iter;
		}
		
		if (DoubleCompare.lessThan(mag, summary.minGradientMagnitude)) {
			summary.minGradientMagnitude = mag;
			summary.minGradientMagnitudeIteration = iter;
			iterationsSinceGradientMagnitudeDecreased.add(0);
		} else {
			int newIterSinceImpr = iterationsSinceGradientMagnitudeDecreased.get(iter-1)+1;
			iterationsSinceGradientMagnitudeDecreased.add(newIterSinceImpr);
		}
		isTimeToStopBasedOnGradientMagnitude();
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
		isTimeToStopBasedOnTrainingError();
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
		isTimeToStopBasedOnValidationError();
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
		if (summary.trainingStoppingIteration == -1 && iterationsSinceTrainingErrorImproved.get(lastIteration) == 20000) {
			System.out.println(String.format("Reached training error stopping criterion after %d iterations.", 
					trainingErrorByIteration.size()));
			summary.trainingStoppingIteration = summary.minTrainingErrorIteration;
			summary.trainingStoppingTrainingError =  trainingErrorByIteration.get((int)summary.minTrainingErrorIteration);
			summary.trainingStoppingValidationError = validationErrorByIteration.get((int)summary.minTrainingErrorIteration);
			summary.trainingStoppingTestError = testErrorByIteration.get((int)summary.minTrainingErrorIteration);
			return true;
		}
		return false;
	}
	
	public boolean isTimeToStopBasedOnValidationError() {
		int lastIteration =  validationErrorByIteration.size() - 1;
		if (summary.validationStoppingIteration == -1 && iterationsSinceValidationErrorImproved.get(lastIteration) == 5000) {
			System.out.println(String.format("Reached validation error stopping criterion after %d iterations.", 
					validationErrorByIteration.size()));
			summary.validationStoppingIteration = summary.minValidationErrorIteration;
			summary.validationStoppingTrainingError = trainingErrorByIteration.get((int)summary.minValidationErrorIteration);
			summary.validationStoppingValidationError = validationErrorByIteration.get((int)summary.minValidationErrorIteration);
			summary.validationStoppingTestError = testErrorByIteration.get((int)summary.minValidationErrorIteration);
			return true;
		}
		return false;
	}
	
	public boolean isTimeToStopBasedOnGradientMagnitude() {
		int lastIteration =  gradientMagnitudesByIteration.size() - 1;
		if (summary.gradientStoppingIteration == -1 && iterationsSinceGradientMagnitudeDecreased.get(lastIteration) == 5000) {
			System.out.println(String.format("Reached gradient magnitude stopping criterion after %d iterations.", 
					gradientMagnitudesByIteration.size()));
			summary.gradientStoppingIteration = summary.minValidationErrorIteration;
			summary.gradientStoppingTrainingError = trainingErrorByIteration.get((int)summary.minValidationErrorIteration);
			summary.gradientStoppingValidationError = validationErrorByIteration.get((int)summary.minValidationErrorIteration);
			summary.gradientStoppingTestError = testErrorByIteration.get((int)summary.minValidationErrorIteration);
			return true;
		}
		return false;
	}
	
	public boolean allStoppingConditionsHaveBeenMet() {
		return summary.gradientStoppingIteration != -1 && summary.validationStoppingIteration != -1 && summary.trainingStoppingIteration != -1;
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
					+ "GradientMagnitude\t"
					+ "Weights\n"));
			for (int i = 0; i < summary.actualNumberOfIterations; i++) {
				bw.write(String.format("%d\t%f\t%f\t%f\t%f\t%f\t%s\n", 
						i, 
						timeInSecondsUpToThisPoint.get(i), 
						trainingErrorByIteration.get(i), 
						validationErrorByIteration.get(i),
						testErrorByIteration.get(i), 
						gradientMagnitudesByIteration.get(i),
						ExtraMatrixMethods.convertWeightsToTabSeparatedString(weightsByIteration.get(i))));
			}
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void generateErrorCurveScript() {
		String directory = summary.directory + "ErrorCurves/";
		new File(directory).mkdirs();
		try {
			BufferedWriter mathScript = new BufferedWriter(new PrintWriter(directory + errorCurveScriptFileName));
			BufferedWriter latexCode = new BufferedWriter(new PrintWriter(directory + latexCodeFileName));
			mathScript.write("\n\ntrainingList = ");
			MathematicaListCreator.convertToMathematicaList(trainingErrorByIteration, 10, mathScript);
			mathScript.write("\n\nvalidationList = ");
			MathematicaListCreator.convertToMathematicaList(validationErrorByIteration, 10, mathScript);
			mathScript.write("\n\ntestList = ");
			MathematicaListCreator.convertToMathematicaList(testErrorByIteration, 10, mathScript);
			mathScript.write("\n\ngradientList = ");
			MathematicaListCreator.convertToMathematicaList(gradientMagnitudesByIteration, 10, mathScript);
			mathScript.write("\n\ngradientStop = ");
			mathScript.write(MathematicaListCreator.verticalLineAtXValueWithHeight(summary.gradientStoppingIteration, 1000000));
			mathScript.write("\n\ntrainingStop = ");
			mathScript.write(MathematicaListCreator.verticalLineAtXValueWithHeight(summary.trainingStoppingIteration, 1000000));
			mathScript.write("\n\nvalidationStop = ");
			mathScript.write(MathematicaListCreator.verticalLineAtXValueWithHeight(summary.validationStoppingIteration, 1000000));
			mathScript.write("\n\n");
			//printPlotCode("trainingPlot", "trainingList", "\"Training Error\"", "{Darker[Red], Opacity[0.5]}", mathScript);
			//printPlotCode("trainingPlot", "trainingList, gradientStop, trainingStop, validationStop", "\"Training Error\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"", "{Darker[Blue], Opacity[0.5]}, {Darker[Pink]}, {Darker[Orange]}, {Darker[Magenta]}", mathScript);

			//printPlotCode("validationPlot", "validationList", "\"Validation Error\"", "{Darker[Blue], Opacity[0.5]}", mathScript);
			//printPlotCode("validationPlot", "validationList, gradientStop, trainingStop, validationStop", "\"Validation Error\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"", "{Darker[Blue], Opacity[0.5]}, {Darker[Pink]}, {Darker[Orange]}, {Darker[Magenta]}", mathScript);
			//printPlotCode("testPlot", "testList, gradientStop, trainingStop, validationStop", "\"Test Error\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"", "{Darker[Green], Opacity[0.5]}, {Darker[Pink]}, {Darker[Orange]}, {Darker[Magenta]}", mathScript);
			printPlotCode("gradientPlot", "gradientList, gradientStop, trainingStop, validationStop", "\"Gradient Magnitude\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"", "{Green}, {Pink}, {Orange}, {Cyan}", mathScript, "" + summary.maxGradientMagnitude);
			
			
			printPlotCode("allPlots", "trainingList, validationList, testList, gradientStop, trainingStop, validationStop", 
					"\"Training Error\", \"Validation Error\", \"Test Error\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"", 
					"{Darker[Red], Opacity[0.5]}, {Darker[Blue], Opacity[0.5]}, {Darker[Green], Opacity[0.5]}, {Pink}, {Orange}, {Cyan}", mathScript);

			//printCombinedPlotCode("allPlots", mathScript, "trainingPlot", "validationPlot", "testPlot");
			
			//printSaveToFileCode("trainingPlot", directory, "trainingPlot", mathScript);
			//printSaveToFileCode("validationPlot", directory, "validationPlot", mathScript);
			//printSaveToFileCode("testPlot", directory, "testPlot", mathScript);
			printSaveToFileCode("gradientPlot", directory, "gradientPlot", mathScript);
			printSaveToFileCode("allPlots", directory, "allPlots", mathScript);
			
			printLatexCode(directory, 
					"trainingPlot", 
					"Training Error: " + summary.parameters.getGradientDescentLatexCaptionPrefix(),  
					"TrainingError" + summary.parameters.getGradientDescentLatexFigureIdPrefix(), 
					latexCode);
			printLatexCode(directory, 
					"validationPlot", 
					"Validation Error: " + summary.parameters.getGradientDescentLatexCaptionPrefix(),  
					"ValidationError" + summary.parameters.getGradientDescentLatexFigureIdPrefix(), 
					latexCode);
			printLatexCode(directory, 
					"testPlot", 
					"Test Error: " + summary.parameters.getGradientDescentLatexCaptionPrefix(),  
					"TestError" + summary.parameters.getGradientDescentLatexFigureIdPrefix(), 
					latexCode);
			printLatexCode(directory, 
					"gradientPlot", 
					"Gradient Magnitudes: " + summary.parameters.getGradientDescentLatexCaptionPrefix(),  
					"GradientMagnitudes" + summary.parameters.getGradientDescentLatexFigureIdPrefix(), 
					latexCode);
			printLatexCode(directory, 
					"allPlots", 
					"All Error Curves: " + summary.parameters.getGradientDescentLatexCaptionPrefix(),  
					"AllErrorCurves" + summary.parameters.getGradientDescentLatexFigureIdPrefix(), 
					latexCode);
			
			latexCode.flush();
			latexCode.close();
			mathScript.flush();
			mathScript.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	public void executeErrorCurveScript() {
		StopWatch timer = new StopWatch().start();
		String directory = summary.directory + "ErrorCurves/";
		try {
			CommandLineExecutor.runProgramAndWaitForItToComplete(directory, new String[] {"cmd", "/c", "math.exe", "-script", errorCurveScriptFileName });
		
			printStatusMessage("Executed error curve script", timer);
		} catch (Exception e) {
			System.err.println(StopWatch.getDateTimeStamp());
			e.printStackTrace();
		}
	}
	
	public void printPlotCode(String plotVariableName, String dataVariableName, String plotLegend, String color, BufferedWriter bw, String yMax) throws IOException {
		bw.write(plotVariableName + " = ListLinePlot[{" + dataVariableName +"}"
				+ ", PlotLegends -> {" + plotLegend + "}"
				+ ", PlotStyle -> {" + color + "}"
				+ ", AxesLabel->{\"Iterations\", \"RMSE\"}"
				+ ", PlotRange -> {{Automatic, Automatic}, {0, " + "Automatic" + "}}"
				+ ", ImageSize -> Large"
				+ "]\n\n");
	}
	
	public void printPlotCode(String plotVariableName, String dataVariableName, String plotLegend, String color, BufferedWriter bw) throws IOException {
		printPlotCode(plotVariableName, dataVariableName, plotLegend, color, bw, "" + (Math.max(summary.maxTestError, Math.max(summary.maxTrainingError, summary.maxValidationError))));
	}
	
	public void printCombinedPlotCode(String plotVariableName, BufferedWriter bw, String... plotVariableNames) throws IOException {
		StringBuilder sb = new StringBuilder();
		for (String s : plotVariableNames) {
			if (sb.length() > 0) {
				sb.append(", ");
			}
			sb.append(s);
		}
		bw.write(plotVariableName + " = Show[" + sb.toString() + ", PlotRange -> All]\n\n");
	}
	
	public void printLatexCode(String directory, String fileNameNoExtension, String caption, String figureId, BufferedWriter bw) throws IOException {
		bw.append("\\begin{figure}[!htb]\\centering\n");
		bw.append("\\includegraphics[width=1\\textwidth]{{" + directory +fileNameNoExtension + "}.png}\n");
		bw.append("\\caption{" + caption + "}\n");
		bw.append("\\label{fig:" +  figureId  + "}\n");
		bw.append("\\end{figure}\n\n");
	}
	
	public void printSaveToFileCode(String plotVariableName, String directory, String fileNameNoExtension, BufferedWriter bw) throws IOException {
		bw.append("fileName = \"" + (directory + fileNameNoExtension)  + "\"\n");
		bw.append("Export[fileName <> \".png\", " + plotVariableName + ", ImageResolution -> 300]\n\n");
	}
	
	public static GradientDescentInformation readFromFile(GradientDescentParameters parameters) {	
		GradientDescentInformation retval = new GradientDescentInformation(parameters);
		retval.summary = GradientDescentSummary.readFromFile(parameters);
		try {
			BufferedReader br = new BufferedReader(new FileReader(retval.summary.directory + retval.fileName));
			br.readLine(); // Skip the header 
			for (int i = 0; i < retval.summary.actualNumberOfIterations; i++) {
				String[] components = br.readLine().split("\t");
				retval.timeInSecondsUpToThisPoint.add(Double.parseDouble(components[1]));
				retval.addTrainingError(Double.parseDouble(components[2]));
				retval.addValidationError(Double.parseDouble(components[3]));
				retval.addTestError(Double.parseDouble(components[4]));
				retval.addGradientMagnitude(Double.parseDouble(components[5]));
				double[] weights = new double[parameters.dataset.numberOfPredictorsPlus1];
				for (int j = 0; j < weights.length; j++) {
					weights[j] = Double.parseDouble(components[6+j]);
				}
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return retval;
	}
	
	public GradientDescentInformation averageInformation(GradientDescentParameters parameters, GradientDescentInformation... infos) {
		GradientDescentInformation retval = new GradientDescentInformation(parameters);
		GradientDescentSummary[] summaries = new GradientDescentSummary[infos.length];
		for (int i = 0; i < infos.length; i++) {
			summaries[i] = infos[i].summary;
		}
		retval.summary = GradientDescentSummary.averageSummary(parameters, summaries);
		SumCountAverage avgTrainingError = new SumCountAverage(), avgValidationError = new SumCountAverage(), avgTestError = new SumCountAverage(), avgGradientMagnitude = new SumCountAverage();
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
					avgGradientMagnitude.addData(info.gradientMagnitudesByIteration.get(i));
					avgWeights.plusEquals(info.weightsByIteration.get(i));
				}
				retval.addTrainingError(avgTrainingError.getMean());
				retval.addValidationError(avgValidationError.getMean());
				retval.addTestError(avgTestError.getMean());
				retval.addGradientMagnitude(avgGradientMagnitude.getMean());
				retval.weightsByIteration.add(avgWeights.timesEquals(1.0 / infos.length));
			}
		}

		return retval;
	}
}
