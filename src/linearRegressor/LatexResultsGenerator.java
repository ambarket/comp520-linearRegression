package linearRegressor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.TreeSet;

import dataset.Dataset;
import dataset.DatasetParameters;
import linearRegressor.GradientDescentSummaryGraphGenerator.GraphableProperty;
import utilities.StopWatch;

public class LatexResultsGenerator {

	public static void writeEntireResultsSection(DatasetParameters dsParam) {
		//tuningParameters.datasets = new DatasetParameters[] {ParameterTuningParameters.crimeCommunitiesParameters};
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(Main.RESULTS_DIRECTORY + dsParam.minimalName + "/entireResultsSection.tex"));
			
			System.out.println("Writing tables");
			// Write all the tables
			writeBestParametersTable(bw, dsParam);
				
			System.out.println("Writing error and gradient magnitude curves");
			writeBestErrorCurves(bw, dsParam);
			
			System.out.println("Writing run data summary curves");
			//writeAverageRunDataSummaryCurves(bw, dsParam);
			
			bw.flush();
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	//-----------------------------------------TABLES------------------------------------------------------------------
	private static void writeBestParametersTable(BufferedWriter bw, DatasetParameters datasetParameters) throws IOException {
		TreeSet<GradientDescentSummary> records = GradientDescentSummary.readGradientDescentSummaries(datasetParameters);

		bw.write("\\begin{table}[!t]\n");
		bw.write("\\centering\n");
		bw.write("\t\\begin{tabular}{  l | c  c  c  }\n");
		
		Comparator<GradientDescentSummary> validationStopComparator = new GradientDescentSummary.ValidationStopComparator();
		Comparator<GradientDescentSummary> gradientStopComparator = new GradientDescentSummary.GradientStopComparator();
		bw.write("\t\t\\multicolumn{4}{c}{} \\\\ \n" + "\\hline \n");
		bw.write(getBestParametersRow(RowType.UpdateRule, records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.LearningRate, records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.Lambda, records, validationStopComparator));
		
	//bw.write(getBestParametersRow(RowType.TimeInSeconds, records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.ValidationStopIterations, records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.ValidationStopTrainingError,records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.ValidationStopValidationError, records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.ValidationStopTestError, records, validationStopComparator));
		
		bw.write(getBestParametersRow(RowType.GradientStopIterations,records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.GradientStopTrainingError, records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.GradientStopValidationError, records, validationStopComparator));
		bw.write(getBestParametersRow(RowType.GradientStopTestError,records, validationStopComparator));
		
		bw.write("\t\t\\multicolumn{4}{c}{} \\\\ \n" + "\\hline \n");

		bw.write(getBestParametersRow(RowType.UpdateRule, records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.LearningRate,records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.Lambda,records, gradientStopComparator));
		
		//bw.write(getBestParametersRow(RowType.TimeInSeconds, records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.ValidationStopIterations, records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.ValidationStopTrainingError, records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.ValidationStopValidationError, records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.ValidationStopTestError, records, gradientStopComparator));
		
		bw.write(getBestParametersRow(RowType.GradientStopIterations, records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.GradientStopTrainingError,records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.GradientStopValidationError, records, gradientStopComparator));
		bw.write(getBestParametersRow(RowType.GradientStopTestError, records, gradientStopComparator));
		
		bw.write("\t\\end{tabular}\n");
		bw.write("\t\\caption{Parameters with Lowest Validation Stop Validation Error (Top) and Lowest Gradient Stop Validation Error (Bottom)}\n");
		bw.write("\t\\label{tab:" + datasetParameters.minimalName + "bestParameters" + "}\n");
		bw.write("\\end{table}\n\n\n");
	}
	
	//-----------------------------------------PLOT LEGENDS------------------------------------------------------------------
	private static void writeGradientMagnitudeCurveLegend(BufferedWriter bw, DatasetParameters dsParam) throws IOException {
		String directory = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/";
		bw.append("\t\\resizebox{0.33\\linewidth}{!}{\n");
		bw.append("\t\t\\includegraphics{{" + (directory + "gradientMagnitudeLegend")  + "}.png}\n");
		bw.append("\t}\n");
	}
	
	private static void writeErrorCurveLegend(BufferedWriter bw, DatasetParameters dsParam) throws IOException {
		String directory = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/";
		bw.append("\t\\resizebox{0.28\\linewidth}{!}{\n");
		bw.append("\t\t\\includegraphics{{" + (directory + "errorCurveLegend")  + "}.png}\n");
		bw.append("\t}\n");
	}
	
	private static void writeLearningCurveLegend(BufferedWriter bw, DatasetParameters dsParam) throws IOException {
		String directory = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescentLearningCurve/";
		bw.append("\t\\resizebox{0.27\\linewidth}{!}{\n");
		bw.append("\t\t\\includegraphics{{" + (directory + "learningCurveLegend")  + "}.png}\n");
		bw.append("\t}\n");
	}
	
	private static void writeRunDataSummaryCurveLegend(BufferedWriter bw, DatasetParameters dsParam) throws IOException {
		String directory = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/";
		bw.append("\t\\resizebox{0.5\\linewidth}{!}{\n");
		bw.append("\t\t\\includegraphics{{" + (directory + "runDataSummaryGraphLegend")  + "}.png}\n");
		bw.append("\t}\n");
	}
	
	//-----------------------------------------PLOTS------------------------------------------------------------------
	private static void writeBestErrorCurves(BufferedWriter bw, DatasetParameters dsParam) throws IOException {
		TreeSet<GradientDescentSummary> records = GradientDescentSummary.readGradientDescentSummaries(dsParam);
		
		TreeSet<GradientDescentSummary> validationStopRecords = new TreeSet<>(new GradientDescentSummary.ValidationStopComparator());
		validationStopRecords.addAll(records);
		
		TreeSet<GradientDescentSummary> gradientStopRecords = new TreeSet<>(new GradientDescentSummary.GradientStopComparator());
		gradientStopRecords.addAll(records);
		
		bw.append("\\begin{figure}[!htb]\\centering\n");
		writeErrorCurveLegend(bw, dsParam);
		writeGradientMagnitudeCurveLegend(bw, dsParam);
		writeLearningCurveLegend(bw, dsParam);
		writeBestErrorCurvesForProvidedRecords(bw, dsParam, validationStopRecords);
		bw.write("\t\\caption{Error (Left) , Gradient Magnitude (Middle), and Learning (Right) Curves for "
				+ "Best Original (Top), Descending Learning Rate (Middle), and Gradient Magnitude Scaling (Bottom) "
				+ "Parameters with Lowest Validation Stop Validation Error" + "}\n");
		bw.write("\t\\label{fig:validStopErrorCurves" + "}\n");
		bw.append("\\end{figure}\n\n");
		
		bw.append("\\begin{figure}[!htb]\\centering\n");
		writeErrorCurveLegend(bw, dsParam);
		writeGradientMagnitudeCurveLegend(bw, dsParam);
		writeLearningCurveLegend(bw, dsParam);
		writeBestErrorCurvesForProvidedRecords(bw, dsParam, gradientStopRecords);
		bw.write("\t\\caption{Error (Left) , Gradient Magnitude (Middle), and Learning (Right) Curves for "
				+ "Best Original (Top), Descending Learning Rate (Middle), and Gradient Magnitude Scaling (Bottom) "
				+ "Parameters with Lowest Gradient Stop Validation Error" + "}\n");
		bw.write("\t\\label{fig:gradientStopErrorCurves" + "}\n");
		bw.append("\\end{figure}\n\n");
	}
	
	private static void writeBestErrorCurvesForProvidedRecords(BufferedWriter bw, DatasetParameters datasetParams, TreeSet<GradientDescentSummary> records ) throws IOException {
		boolean originalErrorCurveDone = false, adaptedLrErrorCurveDone = false, gradientMagErrorCurveDone = false;
		StringBuilder original = new StringBuilder(), adaptedLr = new StringBuilder(), gradientMag = new StringBuilder();
		for (GradientDescentSummary record : records) {
			String directory = Main.RESULTS_DIRECTORY + record.parameters.subDirectory;
			
			String learningCurveFileName = "gradientDescentLearningCurve-" + String.format("%s-%fLR-%fLambda-%dTrainingExamples", 
					record.parameters.updateRule.name(), 
					record.parameters.learningRate, 
					record.parameters.lambda,
					0);
			
			if (!originalErrorCurveDone && record.parameters.updateRule == UpdateRule.Original) {
				original.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				original.append("\t\t\\includegraphics{{" + (directory + "ErrorCurves/allPlots")  + "}.png}\n");
				original.append("\t}\n");
				original.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				original.append("\t\t\\includegraphics{{" + (directory + "ErrorCurves/gradientPlot")  + "}.png}\n");
				original.append("\t}\n");
				original.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				original.append("\t\t\\includegraphics{{" + (Main.RESULTS_DIRECTORY + datasetParams.minimalName + "/gradientDescentLearningCurve/" + learningCurveFileName)  + "}.png}\n");
				original.append("\t}\n");
				
				originalErrorCurveDone = true;
			}
			if (!adaptedLrErrorCurveDone && record.parameters.updateRule == UpdateRule.AdaptedLR) {
				adaptedLr.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				adaptedLr.append("\t\t\\includegraphics{{" + (directory + "ErrorCurves/allPlots")  + "}.png}\n");
				adaptedLr.append("\t}\n");
				adaptedLr.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				adaptedLr.append("\t\t\\includegraphics{{" + (directory + "ErrorCurves/gradientPlot")  + "}.png}\n");
				adaptedLr.append("\t}\n");
				adaptedLr.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				adaptedLr.append("\t\t\\includegraphics{{" + (Main.RESULTS_DIRECTORY + datasetParams.minimalName + "/gradientDescentLearningCurve/" + learningCurveFileName)  + "}.png}\n");
				adaptedLr.append("\t}\n");
				
				adaptedLrErrorCurveDone = true;
			}
			if (!gradientMagErrorCurveDone && record.parameters.updateRule == UpdateRule.GradientMag) {
				gradientMag.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				gradientMag.append("\t\t\\includegraphics{{" + (directory + "ErrorCurves/allPlots")  + "}.png}\n");
				gradientMag.append("\t}\n");
				gradientMag.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				gradientMag.append("\t\t\\includegraphics{{" + (directory + "ErrorCurves/gradientPlot")  + "}.png}\n");
				gradientMag.append("\t}\n");
				gradientMag.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				gradientMag.append("\t\t\\includegraphics{{" + (Main.RESULTS_DIRECTORY + datasetParams.minimalName + "/gradientDescentLearningCurve/" + learningCurveFileName)  + "}.png}\n");
				gradientMag.append("\t}\n");
				gradientMagErrorCurveDone = true;
			}
			if (originalErrorCurveDone && adaptedLrErrorCurveDone && gradientMagErrorCurveDone) {
				break;
			}
		}		
		bw.append(original.toString() + adaptedLr.toString() + gradientMag.toString());
	}

	private static void writeAverageRunDataSummaryCurves(BufferedWriter bw, DatasetParameters dsParam) throws IOException {
		String topDirectory = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/avgSummaryGraphs/";
		GraphableProperty[] xAxes = GradientDescentSummaryGraphGenerator.getXAxes();
		GraphableProperty[] yAxes = GradientDescentSummaryGraphGenerator.getYAxes();
		
		for (GraphableProperty x : xAxes) {
	
			bw.append("\\begin{figure}[!htb]\\centering\n");
			writeRunDataSummaryCurveLegend(bw, dsParam);
			for (GraphableProperty y : yAxes) {
				String directory = topDirectory + GradientDescentSummaryGraphGenerator.convertGraphablePropertyAxesArrayToMinimalString(new GraphableProperty[] {x, y}) + "/";

				/*
				bw.append("\t\\resizebox{0.49\\linewidth}{!}{\n");
				bw.append("\t\t\\includegraphics{{" + (directory + "AllPoints" + GradientDescentSummaryGraphGenerator.convertGraphablePropertyAxesArrayToMinimalString(new GraphableProperty[] {x, y}))  + "}.png}\n");
				bw.append("\t}\n");
				*/
				bw.append("\t\\resizebox{0.32\\linewidth}{!}{\n");
				bw.append("\t\t\\includegraphics{{" + (directory + "UniquePoints" + GradientDescentSummaryGraphGenerator.convertGraphablePropertyAxesArrayToMinimalString(new GraphableProperty[] {x, y}))  + "}.png}\n");
				bw.append("\t}\n");
			}		
			bw.write("\t\\caption{" + x.toString() + " vs. Errors and Stopping Conditions}\n");
			bw.write("\t\\label{fig:" + x.name() + "VsAll}\n");
			bw.append("\\end{figure}\n\n");
		}

	}
	
	//-----------------------------------------HELPERS------------------------------------------------------------------
	private enum RowType {
		TimeInSeconds("RunningTime (seconds)"), ValidationStopIterations("Validation Stop Iteration"), GradientStopIterations("Gradient Stop Iteration"), 
		ValidationStopTrainingError("Validation Stop Training Error"), ValidationStopValidationError("Validation Stop Validation Error"), ValidationStopTestError("Validation Stop Test Error"),
		GradientStopTrainingError("Gradient Stop Training Error"), GradientStopValidationError("Gradient Stop Validation Error"),  GradientStopTestError("Gradient Stop Test Error"),
		UpdateRule("Update Rule"), LearningRate("Learning Rate"), Lambda("Lambda");
		
	    private final String fieldDescription;

	    private RowType() {
	        fieldDescription = null;
	    }
	    
	    private RowType(String value) {
	        fieldDescription = value;
	    }

	    @Override
	    public String toString() {
	        return (fieldDescription == null) ? name() : fieldDescription;
	    }
	    
	}
	
	/**
	 * Find n best constant and n best variable and % improvement using variable
	 * @param rowType
	 * @param records
	 * @param n
	 * @return
	 */
	private static String getBestParametersRow(RowType rowType, TreeSet<GradientDescentSummary> records, Comparator<GradientDescentSummary> comparator) {
		StringBuilder retval = new StringBuilder();
		retval.append("\t\t\\rule{0pt}{2ex} " + rowType + " & ");
		double[] values = new double[3];
		String[] formattedValues = new String[3];
		
		TreeSet<GradientDescentSummary> originalRecords = new TreeSet<>(comparator); 
		originalRecords.addAll(GradientDescentSummaryFilter.updateRuleEqualsOriginal.filterRecordsOnParameterValue(records));
		
		TreeSet<GradientDescentSummary> adaptedLrRecords = new TreeSet<>(comparator); 
		adaptedLrRecords.addAll(GradientDescentSummaryFilter.updateRuleEqualsAdaptedLR.filterRecordsOnParameterValue(records));
	
		TreeSet<GradientDescentSummary> gradientMagRecords = new TreeSet<>(comparator); 
		gradientMagRecords.addAll( GradientDescentSummaryFilter.updateRuleEqualsGradientMag.filterRecordsOnParameterValue(records));
		
		for (int i = 0; i < 3; i++) {
			GradientDescentSummary record = null;
			switch (i) {
				case 0:
					record = originalRecords.first();
					break;
				case 1:
					record = adaptedLrRecords.first();
					break;
				case 2:
					record = gradientMagRecords.first();
					break;
			}

			switch (rowType) {
				case UpdateRule:
					values[i] = Double.NaN;
					formattedValues[i] = record.parameters.updateRule.toString();
					break;
				case LearningRate:
					values[i] = record.parameters.learningRate;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
				case Lambda:
					values[i] = record.parameters.lambda;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
	

				case TimeInSeconds:
					values[i] = record.timeInSeconds;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
					
				case ValidationStopIterations:
					values[i] = record.validationStoppingIteration;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
				case ValidationStopTrainingError:
					values[i] = record.validationStoppingTrainingError;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
				case ValidationStopValidationError:
					values[i] = record.validationStoppingValidationError;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
				case ValidationStopTestError:
					values[i] = record.validationStoppingTestError;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;


				case GradientStopIterations:
					values[i] = record.gradientStoppingIteration;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
				case GradientStopTrainingError:
					values[i] = record.gradientStoppingTrainingError;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
				case GradientStopValidationError:
					values[i] = record.gradientStoppingValidationError;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
				case GradientStopTestError:
					values[i] = record.gradientStoppingTestError;
					formattedValues[i] = (String.format("%.6f", values[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1");
					break;
				default:
					break;
			}
		}
		int minIndex = Integer.MAX_VALUE;
		if (rowType != RowType.UpdateRule && rowType != RowType.Lambda && rowType != RowType.LearningRate) {
			minIndex = getIndexWithMin(values);
		}
		for (int j = 0; j < formattedValues.length; j++) {
			if (j == minIndex) {
				formattedValues[j] = "\\textbf{" + formattedValues[j] + "}";
			}
			
			retval.append(formattedValues[j]);
			if (j != formattedValues.length-1) {
				retval.append(" & ");
			}
		}
		return retval.toString() + "\\\\ \\hline \n ";
	}

	
	private static int getIndexWithMin(double[] array) {
		int minIndex = 0;
		for (int i = 1; i < array.length; i++){
		   double newnumber = array[i];
		   if ((newnumber < array[minIndex])){
			   minIndex = i;
		  }
		} 
		return minIndex;
	}
}
