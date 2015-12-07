package linearRegressor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import utilities.CommandLineExecutor;
import utilities.ExtraMatrixMethods;
import utilities.MathematicaListCreator;
import utilities.StopWatch;
import utilities.SumCountAverage;
import Jama.Matrix;
import dataset.DatasetParameters;

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
		
		if (summary.maxGradientMagnitude < mag) {
			summary.maxGradientMagnitude = mag;
			summary.maxGradientMagnitudeIteration = iter;
		}
		
		if (mag < summary.minGradientMagnitude) {
			summary.minGradientMagnitude = mag;
			summary.minGradientMagnitudeIteration = iter;
			iterationsSinceGradientMagnitudeDecreased.add(0);
		} else {
			int newIterSinceImpr = iterationsSinceGradientMagnitudeDecreased.get(iter-1)+1;
			iterationsSinceGradientMagnitudeDecreased.add(newIterSinceImpr);
		}
		isTimeToStopBasedOnGradientMagnitude(false);
	}
	
	public void addTrainingError(double error) {
		trainingErrorByIteration.add(error);
		int iter = trainingErrorByIteration.size() - 1;
		if (summary.maxTrainingError <error) {
			summary.maxTrainingError = error;
			summary.maxTrainingErrorIteration = iter;
		}
		
		if (error < summary.minTrainingError) {
			summary.minTrainingError = error;
			summary.minTrainingErrorIteration = iter;
			iterationsSinceTrainingErrorImproved.add(0);
		} else {
			int newIterSinceImpr = iterationsSinceTrainingErrorImproved.get(iter-1)+1;
			iterationsSinceTrainingErrorImproved.add(newIterSinceImpr);
		}
		isTimeToStopBasedOnTrainingError(false);
	}
	
	public void addValidationError(double error) {
		validationErrorByIteration.add(error);
		int iter = validationErrorByIteration.size() - 1;
		if (summary.maxValidationError < error) {
			summary.maxValidationError = error;
			summary.maxValidationErrorIteration = iter;
		}
		
		if (error < summary.minValidationError) {
			summary.minValidationError = error;
			summary.minValidationErrorIteration = iter;
			iterationsSinceValidationErrorImproved.add(0);
		} else {
			int newIterSinceImpr = iterationsSinceValidationErrorImproved.get(iter-1)+1;
			iterationsSinceValidationErrorImproved.add(newIterSinceImpr);
		}
		isTimeToStopBasedOnValidationError(false);
	}
	
	public void addTestError(double error) {
		testErrorByIteration.add(error);
		int iter = testErrorByIteration.size() - 1;
		if (summary.maxTestError < error) {
			summary.maxTestError = error;
			summary.maxTestErrorIteration = iter;
		}
		
		if (error < summary.minTestError) {
			summary.minTestError = error;
			summary.minTestErrorIteration = iter;
		}
	}
	
	public boolean isTimeToStopBasedOnTrainingError(boolean force) {
		int lastIteration =  trainingErrorByIteration.size() - 1;
		if (summary.trainingStoppingIteration == -1 && (force | iterationsSinceTrainingErrorImproved.get(lastIteration) == 20000)) {
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
	
	public boolean isTimeToStopBasedOnValidationError(boolean force) {
		int lastIteration =  validationErrorByIteration.size() - 1;
		if (summary.validationStoppingIteration == -1 && (force | iterationsSinceValidationErrorImproved.get(lastIteration) == 5000)) {
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
	
	public boolean isTimeToStopBasedOnGradientMagnitude(boolean force) {
		int lastIteration =  gradientMagnitudesByIteration.size() - 1;
		if (summary.gradientStoppingIteration == -1 && (force | iterationsSinceGradientMagnitudeDecreased.get(lastIteration) == 5000)) {
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
					+ "GradientMagnitude\n"));
			for (int i = 0; i < summary.actualNumberOfIterations; i++) {
				bw.write(String.format("%d\t%f\t%f\t%f\t%f\t%f\n", 
						i, 
						timeInSecondsUpToThisPoint.get(i), 
						trainingErrorByIteration.get(i), 
						validationErrorByIteration.get(i),
						testErrorByIteration.get(i), 
						gradientMagnitudesByIteration.get(i)
						/*ExtraMatrixMethods.convertWeightsToTabSeparatedString(weightsByIteration.get(i))*/));
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
			MathematicaListCreator.convertToMathematicaList(trainingErrorByIteration, 5, mathScript);
			mathScript.write("\n\nvalidationList = ");
			MathematicaListCreator.convertToMathematicaList(validationErrorByIteration, 5, mathScript);
			mathScript.write("\n\ntestList = ");
			MathematicaListCreator.convertToMathematicaList(testErrorByIteration, 5, mathScript);
			mathScript.write("\n\ngradientList = ");
			MathematicaListCreator.convertToMathematicaList(gradientMagnitudesByIteration, 5, mathScript);
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
			printPlotCode("gradientPlot", "gradientList, gradientStop, trainingStop, validationStop", "\"Gradient Magnitude\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"", "{Green}, {Pink}, {Orange}, {Blue}", mathScript, "" + summary.maxGradientMagnitude);
			
			
			printPlotCode("allPlots", "trainingList, validationList, testList, gradientStop, trainingStop, validationStop", 
					"\"Training Error\", \"Validation Error\", \"Test Error\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"", 
					"{Red, Opacity[0.85]}, {Cyan, Opacity[0.85]}, {Black, Opacity[0.85]}, {Pink}, {Orange}, {Blue}", mathScript);

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
			
			CommandLineExecutor.executeMathematicaScript(directory, errorCurveScriptFileName);
		
			printStatusMessage("Executed error curve script", timer);
		} catch (Exception e) {
			System.err.println(StopWatch.getDateTimeStamp());
			e.printStackTrace();
		}
	}
	
	public void printPlotCode(String plotVariableName, String dataVariableName, String plotLegend, String color, BufferedWriter bw, String yMax) throws IOException {
		String frameLabel = "";
		if (plotVariableName.contains("gradientPlot")) {
			frameLabel = "FrameLabel->{\"Iterations\", \"Gradient Magnitude\"}";
		} else {
			frameLabel = "FrameLabel->{\"Iterations\", \"RMSE\"}";
		}
		bw.write(plotVariableName + " = ListLinePlot[{" + dataVariableName +"}"
				//+ ", PlotLegends -> {" + plotLegend + "}"
				+ ", PlotStyle -> {" + color + "}"
				//+ ", AxesLabel->{\"Iterations\", \"RMSE\"}"
				+ ", PlotRange -> {{Automatic, Automatic}, {0, " + "Automatic" + "}}"
				+ ", ImageSize -> Large"
				+ ", Frame->True, FrameStyle->Black , FrameTicksStyle->Black, LabelStyle->{Black, 12}, " + frameLabel
				+ ", PlotRangePadding->{{Scaled[0.03],Scaled[0.03]}, {Scaled[0.03], Scaled[0.03]}}"
				+ ", ImageMargins->{{0,0},{5,5}}"
				+ "]\n\n");
	}
	
	public static void generateAndExecutePlotLegend(DatasetParameters dsParam) {
		String gradientMagfile = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/gradientMagnitudeLegend";
		String errorCurveFile = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/errorCurveLegend";
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(gradientMagfile + ".m"));
			bw.append("gradientMagnitudeLegend = LineLegend[{Green, Pink, Orange, Blue}, {\"Gradient Magnitude\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"}]\n\n");
			bw.append("fileName = \"" + gradientMagfile  + "\"\n");
			bw.append("Export[fileName <> \".png\", gradientMagnitudeLegend, ImageResolution -> 300]\n\n");
			bw.flush();
			bw.close();
			bw = new BufferedWriter(new PrintWriter(errorCurveFile + ".m"));
			bw.append("errorCurveLegend = LineLegend[{Red, Cyan, Black, Pink, Orange, Blue}, {\"Training Error\", \"Validation Error\", \"Test Error\", \"Gradient Stop\", \"Training Stop\", \"Validation Stop\"}]\n\n");
			bw.append("fileName = \"" + errorCurveFile  + "\"\n");
			bw.append("Export[fileName <> \".png\", errorCurveLegend, ImageResolution -> 300]\n\n");
			bw.flush();
			bw.close();
			CommandLineExecutor.executeMathematicaScript(Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/", "gradientMagnitudeLegend.m");
			CommandLineExecutor.executeMathematicaScript(Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/", "errorCurveLegend.m");
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

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
		retval.summary = new GradientDescentSummary(parameters);//.readFromFile(parameters);
		int i = 0;
		try {
			BufferedReader br = new BufferedReader(new FileReader(retval.summary.directory + retval.fileName));
			br.readLine(); // Skip the header 
			String line = null;
			
			while ((line = br.readLine()) != null) {
			//for (int i = 0; i < retval.summary.actualNumberOfIterations; i++) {
				String[] components = line.split("\t");
				retval.timeInSecondsUpToThisPoint.add(Double.parseDouble(components[1]));
				retval.addTrainingError(Double.parseDouble(components[2]));
				retval.addValidationError(Double.parseDouble(components[3]));
				retval.addTestError(Double.parseDouble(components[4]));
				retval.addGradientMagnitude(Double.parseDouble(components[5]));
				//double[] weights = new double[parameters.dataset.numberOfPredictorsPlus1];
				//for (int j = 0; j < weights.length; j++) {
				//	weights[j] = Double.parseDouble(components[6+j]);
				//}
				i++;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		retval.summary.actualNumberOfIterations = i;
		retval.summary.timeInSeconds = retval.timeInSecondsUpToThisPoint.get(i-1);
		retval.isTimeToStopBasedOnGradientMagnitude(true);
		retval.isTimeToStopBasedOnTrainingError(true);
		retval.isTimeToStopBasedOnValidationError(true);
		return retval;
	}
	
	public static GradientDescentInformation averageInformation(GradientDescentParameters parameters, GradientDescentInformation... infos) {
		GradientDescentInformation retval = new GradientDescentInformation(parameters);
		GradientDescentSummary[] summaries = new GradientDescentSummary[infos.length];
		for (int i = 0; i < infos.length; i++) {
			summaries[i] = infos[i].summary;
		}
		retval.summary = GradientDescentSummary.averageSummary(parameters, summaries);
		SumCountAverage avgTrainingError = new SumCountAverage(), avgValidationError = new SumCountAverage(), avgTestError = new SumCountAverage(), avgGradientMagnitude = new SumCountAverage(), avgTimeInSeconds = new SumCountAverage();
		Matrix avgWeights = null;

		for (int i =0; i < retval.summary.minimumIterationsAllRunsShare; i++) {
			avgTrainingError.reset();
			avgValidationError.reset();
			avgTestError.reset();
			avgTimeInSeconds.reset();
			
			for (GradientDescentInformation info : infos) {
				avgTrainingError.addData(info.trainingErrorByIteration.get(i));
				avgValidationError.addData(info.validationErrorByIteration.get(i));
				avgTestError.addData(info.testErrorByIteration.get(i));
				avgGradientMagnitude.addData(info.gradientMagnitudesByIteration.get(i));
				avgTimeInSeconds.addData(info.timeInSecondsUpToThisPoint.get(i));
				//avgWeights.plusEquals(info.weightsByIteration.get(i));
			}
			//retval.summary.
			retval.addTrainingError(avgTrainingError.getMean());
			retval.addValidationError(avgValidationError.getMean());
			retval.addTestError(avgTestError.getMean());
			retval.timeInSecondsUpToThisPoint.add(avgTimeInSeconds.getMean());
			retval.addGradientMagnitude(avgGradientMagnitude.getMean());
			//retval.weightsByIteration.add(avgWeights.timesEquals(1.0 / infos.length));
		}
		retval.summary.actualNumberOfIterations = retval.summary.minimumIterationsAllRunsShare;
		retval.isTimeToStopBasedOnGradientMagnitude(true);
		retval.isTimeToStopBasedOnTrainingError(true);
		retval.isTimeToStopBasedOnValidationError(true);

		return retval;
	}
}
