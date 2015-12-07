package linearRegressor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import dataset.DatasetParameters;
import utilities.CommandLineExecutor;
import utilities.MaxAndMin;
import utilities.SimpleHostLock;
import utilities.StopWatch;
import utilities.SumCountAverage;

public class GradientDescentSummaryGraphGenerator {
	public enum GraphableProperty {
		TimeInSeconds("Running Time"), 
		gradientStopTrainingError ("Training Error @ Gradient Stop"), 
		gradientStopValidationError ("Validation Error @ Gradient Stop"), 
		gradientStopTestError ("Test Error @ Gradient Stop"), 
		
		ValidationStopTrainingError ("Training Error @ Validation Stop"), 
		ValidationStopValidationError ("Validation Error @ Validation Stop"), 
		ValidationStopTestError ("Test Error @ Validation Stop"), 
		
		GradientStoppingIteration("Gradient Magnitude Stop"), 
		ValidationStoppingIteration("Validation Error Stop"), 
		TrainingStoppingIteration("Training Error Stop"),
		
		Lambda("Lambda"), 
		LearningRate("Learning Rate");
		
		String niceName = null;
		GraphableProperty(String niceName) {
			this.niceName = niceName;
		}
		public String toString() {
			return niceName;
		}
	};
	
	public enum GraphType {UniquePoints, AllPoints};
	
	public static ArrayList<GraphableProperty[]> getAllAxes() {
		GraphableProperty[] yAxes = getYAxes();
	
	
		ArrayList<GraphableProperty[]> graphAxes = new ArrayList<GraphableProperty[]> ();
		
			for (GraphableProperty y : yAxes) {
				graphAxes.addAll(getAxesWithSpecifiedYAxis(y));
		}
		return graphAxes;
	}
	
	public static GraphableProperty[] getXAxes() {
		return new GraphableProperty[] {
				GraphableProperty.Lambda, 
				GraphableProperty.LearningRate};
	}
	
	public static GraphableProperty[] getYAxes() {
		return new GraphableProperty[] {GraphableProperty.TimeInSeconds, 
				GraphableProperty.TrainingStoppingIteration, 
				GraphableProperty.ValidationStoppingIteration, 
				GraphableProperty.GradientStoppingIteration, 

				GraphableProperty.ValidationStopTrainingError, 
				GraphableProperty.ValidationStopValidationError,  
				GraphableProperty.ValidationStopTestError, 
				
				GraphableProperty.gradientStopTrainingError, 
				GraphableProperty.gradientStopValidationError,  
				GraphableProperty.gradientStopTestError, 

				};
	}
	
	public static ArrayList<GraphableProperty[]> getAxesWithSpecifiedYAxis(GraphableProperty y) {
		GraphableProperty[] xAxes = getXAxes();
	
		ArrayList<GraphableProperty[]> graphAxes = new ArrayList<GraphableProperty[]> ();
		for (GraphableProperty x : xAxes) {
			graphAxes.add(new GraphableProperty[] {x, y});
		}
		return graphAxes;
	}
	
	public static void generateAndSaveGraphs(DatasetParameters dsParam) {
		TreeSet<GradientDescentSummary> allRecords = GradientDescentSummary.readGradientDescentSummaries(dsParam);
		
		ExecutorService executor = Executors.newCachedThreadPool();
		
		String outputDirectory = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/avgSummaryGraphs/";

		ArrayList<GraphableProperty[]> graphAxes = getAllAxes();
		int submissionNumber = 0;
		int totalNumberOfTests = graphAxes.size();
		StopWatch globalTimer = new StopWatch().start();
		Queue<Future<Void>> futureQueue = new LinkedList<Future<Void>>();

		for (GraphableProperty[] axes : graphAxes) {
			futureQueue.add(executor.submit(
					new RunDataSummaryGraphTask(dsParam, allRecords, outputDirectory, submissionNumber, totalNumberOfTests, globalTimer, axes)));
			
			if (futureQueue.size() >= 24) {
				System.out.println(StopWatch.getDateTimeStamp() + "Reached 8 run data summary graph threads, waiting for some to finish");
				while (futureQueue.size() > 12) {
					try {
						futureQueue.poll().get();

					} catch (InterruptedException e) {
						System.err.println(StopWatch.getDateTimeStamp());
						e.printStackTrace();
					} catch (ExecutionException e) {
						System.err.println(StopWatch.getDateTimeStamp());
						e.printStackTrace();
					}
				}
			}
		}
		System.out.println(StopWatch.getDateTimeStamp() + "Submitted the last of the run data summary graph jobs, just waiting until they are all done.");
		while (!futureQueue.isEmpty()) {
			try {
				futureQueue.poll().get();
			} catch (InterruptedException e) {
				System.err.println(StopWatch.getDateTimeStamp());
				e.printStackTrace();
			} catch (ExecutionException e) {
				System.err.println(StopWatch.getDateTimeStamp());
				e.printStackTrace();
			}
		}
		System.out.println(StopWatch.getDateTimeStamp() + "Finished generating run data summary graph for all filters and axes.");
		executor.shutdownNow();
	}
	
	private static class RunDataSummaryGraphTask implements Callable<Void>{
		TreeSet<GradientDescentSummary> allRecords; 
		String outputDirectory;
		GraphableProperty[] axes;
		int submissionNumber;
		int totalNumberOfTests;
		StopWatch globalTimer;
		DatasetParameters dsParam;
		
		public RunDataSummaryGraphTask(
				DatasetParameters dsParams,
				TreeSet<GradientDescentSummary> allRecords, 
				String outputDirectory,
				int submissionNumber,
				int totalNumberOfTests,
				StopWatch globalTimer,
				GraphableProperty... axes) {
			this.dsParam = dsParams;
			this.allRecords = allRecords;
			this.outputDirectory = outputDirectory;
			this.axes = axes;
			this.submissionNumber = submissionNumber;
			this.totalNumberOfTests = totalNumberOfTests;
			this.globalTimer = globalTimer;
		}
		
		@Override
		public Void call() {
			StopWatch timer = new StopWatch().start();
			String testSubDirectory = convertGraphablePropertyAxesArrayToMinimalString(axes) + "/";
			String locksDir = Main.LOCKS_DIRECTORY + dsParam.minimalName + "/gradientDescent/avgSummaryGraphs/" + testSubDirectory;
			new File(locksDir).mkdirs();
			if (SimpleHostLock.checkDoneLock(locksDir + "runDataSummaryGraphLock.txt")) {
				System.out.println(StopWatch.getDateTimeStamp() + String.format("[AVERAGED] Already generated run data summary graph for %s (%d out of %d) in %.4f minutes. Have been running for %.4f minutes total."
						, testSubDirectory, submissionNumber, totalNumberOfTests, timer.getElapsedMinutes(), globalTimer.getElapsedMinutes()));
				return null;
			}
			
			
			if (axes.length != 2) {
				System.out.println(StopWatch.getDateTimeStamp() + "Only defined for 2D graphs");
			}
			
			TreeSet<GradientDescentSummary>  originalRecords = GradientDescentSummaryFilter.updateRuleEqualsOriginal.filterRecordsOnParameterValue(allRecords);
			TreeSet<GradientDescentSummary> adaptedLrRecords = GradientDescentSummaryFilter.updateRuleEqualsAdaptedLR.filterRecordsOnParameterValue(allRecords);
			TreeSet<GradientDescentSummary> gradientMagRecords = GradientDescentSummaryFilter.updateRuleEqualsGradientMag.filterRecordsOnParameterValue(allRecords);
			
			
			String originalUniquePointsDataListCode = "",
					adaptedLrUniquePointsDataListCode = "",
					gradientMagUniquePointsDataListCode = "",
					combinedUniquePointsPlotCode = "", 
					
					originalAllPointsDataListCode = "",
					adaptedLrAllPointsDataListCode = "", 
					gradientMagAllPointsDataListCode = "",
					combinedAllPointsPlotCode = "";


			originalUniquePointsDataListCode = getMathematicaDataListCode(UpdateRule.Original, GraphType.UniquePoints, originalRecords, axes);
			originalAllPointsDataListCode = getMathematicaDataListCode(UpdateRule.Original, GraphType.AllPoints, originalRecords, axes);

			adaptedLrUniquePointsDataListCode = getMathematicaDataListCode(UpdateRule.AdaptedLR, GraphType.UniquePoints, adaptedLrRecords, axes);
			adaptedLrAllPointsDataListCode = getMathematicaDataListCode(UpdateRule.AdaptedLR, GraphType.AllPoints, adaptedLrRecords, axes);
			
			gradientMagUniquePointsDataListCode = getMathematicaDataListCode(UpdateRule.GradientMag, GraphType.UniquePoints, gradientMagRecords, axes);
			gradientMagAllPointsDataListCode = getMathematicaDataListCode(UpdateRule.GradientMag, GraphType.AllPoints, gradientMagRecords, axes);

			combinedAllPointsPlotCode = getMathematicaCombinedPlotCode(GraphType.AllPoints, originalRecords, adaptedLrRecords, gradientMagRecords, outputDirectory, axes);
			combinedUniquePointsPlotCode = getMathematicaCombinedPlotCode(GraphType.UniquePoints, originalRecords, adaptedLrRecords, gradientMagRecords, outputDirectory, axes);
			
			String fileDirectory = outputDirectory + convertGraphablePropertyAxesArrayToMinimalString(axes) + "/";
			String mathematicaFilePath = fileDirectory + convertGraphablePropertyAxesArrayToMinimalString(axes) + ".m";

			try {
				new File(fileDirectory).mkdirs();
				BufferedWriter mathematica = new BufferedWriter(new PrintWriter(new File(mathematicaFilePath)));
				mathematica.write(originalUniquePointsDataListCode + "\n" + originalAllPointsDataListCode + "\n\n");
				mathematica.write(adaptedLrUniquePointsDataListCode + "\n" + adaptedLrAllPointsDataListCode + "\n\n");
				mathematica.write(gradientMagUniquePointsDataListCode + "\n" + gradientMagAllPointsDataListCode + "\n\n");
				mathematica.write(combinedUniquePointsPlotCode + "\n" + combinedAllPointsPlotCode + "\n\n");

				mathematica.flush();
				mathematica.close();
			} catch (Exception e) {
				System.err.println(StopWatch.getDateTimeStamp());
				e.printStackTrace();
				System.exit(1);
			}
			
			try {
				StopWatch mathematicaCurveTimer = new StopWatch().start();
				mathematicaCurveTimer.printMessageWithTime("Starting execution of " + mathematicaFilePath);
				CommandLineExecutor.executeMathematicaScript(fileDirectory, convertGraphablePropertyAxesArrayToMinimalString(axes)  + ".m");
				mathematicaCurveTimer.printMessageWithTime("Finished execution of " + mathematicaFilePath);
			} catch (Exception e) {
				System.err.println(StopWatch.getDateTimeStamp());
				e.printStackTrace();
				System.out.println(StopWatch.getDateTimeStamp() + String.format("[AVERAGED] Failed to execute the mathematica code for the run data summary graph for %s, not writing done lock. (%d out of %d) in %.4f minutes. Have been running for %.4f minutes total.", 
						testSubDirectory, submissionNumber, totalNumberOfTests, timer.getElapsedMinutes(), globalTimer.getElapsedMinutes()));
				return null;
			}
			System.out.println(StopWatch.getDateTimeStamp() + String.format("[AVERAGED] Successfully generated the run data summary graph for %s (%d out of %d) in %.4f minutes. Have been running for %.4f minutes total.", 
					testSubDirectory, submissionNumber, totalNumberOfTests, timer.getElapsedMinutes(), globalTimer.getElapsedMinutes()));
			SimpleHostLock.writeDoneLock(locksDir + "runDataSummaryGraphLock.txt");
			return null;
		}
	}
	
	private static String getMathematicaDataListCode( 
			UpdateRule updateRule, 
			GraphType graphType,
			TreeSet<GradientDescentSummary> filteredRecords, 
			GraphableProperty[] axes) {
		
		String dataListVariableName = updateRule.name() + graphType.name() +  convertGraphablePropertyAxesArrayToMinimalString(axes) + "Data";
		TreeSet<Point> points = getPoints(graphType, filteredRecords, axes);
		int numberOfUniquePoints = countNumberOfUniqueXAndYValues(points);
		
		// Theres no value in graphing this.
		if (numberOfUniquePoints <= 1) {
			return null;
		}
		
		StringBuffer buffer = new StringBuffer();
		boolean first = true;
		buffer.append(dataListVariableName + " = {");
		for (Point point : points) {
			if (!first) {
				buffer.append(", ");
			}
			buffer.append(point.getMathematicaListEntry());
			first = false;
		}
		buffer.append("}\n");
		return buffer.toString();
	}

	/** 
	 * Return the minimum of the number of unique x value and unique y values (only if 3D).
	 * There's no point in graphing a 2D graph with only one X value, or a 3D graph with only 1 X or only one Y value
	 * @param points
	 * @return
	 */
	
	private static int countNumberOfUniqueXAndYValues(TreeSet<Point> points) {
		HashSet<Double> xValues = new HashSet<Double>();
		HashSet<Double> yValues = new HashSet<Double>();
		
		boolean threeDimensional = points.first().values.length == 3;
		for (Point point : points) {
			xValues.add(point.values[0]);
		
			if (threeDimensional) {
				yValues.add(point.values[1]);
			}
		}
		int retval = xValues.size();
		if (threeDimensional) {
			retval = Math.min(retval, yValues.size());
		}
		return retval;
	}
	
	private static String getMathematicaCombinedPlotCode(
			GraphType graphType, 
			TreeSet<GradientDescentSummary> originalRecords, 
			TreeSet<GradientDescentSummary> adaptedLrRecords, 
			TreeSet<GradientDescentSummary> gradientMagRecords, 
			String outputDirectory,
			GraphableProperty[] axes) {
		
		String plotRange = getPlotRangeOfCombinedPlot(originalRecords, adaptedLrRecords, gradientMagRecords, axes);
		String ticks = getTicks(axes);
		String frame = getFrame(axes);
		
		String plotVariableName = graphType.name() + convertGraphablePropertyAxesArrayToMinimalString(axes);
		String originalDataListVariableName = UpdateRule.Original.name() + graphType.name() + convertGraphablePropertyAxesArrayToMinimalString(axes) + "Data";
		String adaptedLrDataListVariableName = UpdateRule.AdaptedLR.name() + graphType.name() + convertGraphablePropertyAxesArrayToMinimalString(axes) + "Data";
		String gradientMagDataListVariableName = UpdateRule.GradientMag.name() + graphType.name() + convertGraphablePropertyAxesArrayToMinimalString(axes) + "Data";
		StringBuffer buffer = new StringBuffer();

		String plotCommand = null;
		String extraCommands = "";
		
		if (graphType == GraphType.UniquePoints) {
			plotCommand = "ListLinePlot";
		} else {
			plotCommand = "ListPlot";
		}
		
		String plotRangePadding = "PlotRangePadding->{{Scaled[0.03],Scaled[0.03]}, {Scaled[0.03], Scaled[0.03]}}";
		String imageMargins = "ImageMargins->{{0,0},{5,5}}";
		buffer.append(String.format("%s = %s[%s, %s, %s, %s, %s, %s, %s, %s %s]\n", 
				plotVariableName, 
				plotCommand,
				("{" + originalDataListVariableName + ", " + adaptedLrDataListVariableName + ", " + gradientMagDataListVariableName + "}"), 
				plotRange, 
				ticks, 
				"PlotStyle -> {{Red, Opacity[0.85]}, {Blue, Opacity[0.85]}, {Green, Opacity[0.85]}}",
				"PlotMarkers -> {Automatic, Small}",
				frame,
				plotRangePadding,
				imageMargins,
				extraCommands)
			);
		
		String fileDirectory = outputDirectory + convertGraphablePropertyAxesArrayToMinimalString(axes) + "/";
		String filePath = fileDirectory + graphType.name() + convertGraphablePropertyAxesArrayToMinimalString(axes) + ".png";
	
		buffer.append(String.format("%sFilePath = \"%s\"\n", plotVariableName, filePath));
		buffer.append(String.format("Export[%sFilePath, %s , ImageResolution -> 300]\n\n", plotVariableName, plotVariableName));
		

		return buffer.toString();
	}
	
	public static void generateAndExecutePlotLegend(DatasetParameters dsParam) {
		String file = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/runDataSummaryGraphLegend";
		
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(file + ".m"));
			bw.append("runDataSummaryGraphLegend = LineLegend[{Red, Blue, Green}, {\"Original\", \"Descending Learning Rate\", \"Gradient Magnitude Scaling\"}]\n\n");
			bw.append("fileName = \"" + file  + "\"\n");
			bw.append("Export[fileName <> \".png\", runDataSummaryGraphLegend, ImageResolution -> 300]\n\n");
			bw.flush();
			bw.close();
			CommandLineExecutor.executeMathematicaScript(Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescent/", "runDataSummaryGraphLegend.m");
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

	}
	
	private static TreeSet<Point> getPoints(GraphType graphType, TreeSet<GradientDescentSummary> allRecords, GraphableProperty[] axes) {
		TreeSet<Point> points = new TreeSet<Point>();
		
		if (graphType == GraphType.UniquePoints) {
			HashMap<UniqueXYPointKey, SumCountAverage> uniqueXYPointKeyToAvgZValueMap = new HashMap<>();
			
			for (GradientDescentSummary record : allRecords) {
				UniqueXYPointKey key = new UniqueXYPointKey(record, axes);
				SumCountAverage averageZValue = uniqueXYPointKeyToAvgZValueMap.get(key);
				if (averageZValue == null) {
					averageZValue = new SumCountAverage();
					uniqueXYPointKeyToAvgZValueMap.put(key, averageZValue);
				}
				averageZValue.addData(getPropertyValue(record, axes[axes.length-1]));
			}
			for (Map.Entry<UniqueXYPointKey, SumCountAverage> entry : uniqueXYPointKeyToAvgZValueMap.entrySet()) {
				points.add(new Point(entry.getKey(), entry.getValue()));
			}
		} else if (graphType == GraphType.AllPoints) {
			for (GradientDescentSummary record : allRecords) {
				points.add(new Point(record, axes));
			}
		} else {
			throw new IllegalArgumentException();
		}

		return points;
	}
	
	private static String getFrame(GraphableProperty[] axes) {
		StringBuffer buffer = new StringBuffer();
		buffer.append("Frame->True");
		buffer.append(", FrameStyle->Black");
		buffer.append(", FrameTicksStyle->Black");
		buffer.append(", LabelStyle->{Black, 12}");
		buffer.append(", FrameLabel->{");
		boolean first = true;
		
		for (GraphableProperty property : axes) {
			if (!first) {
				buffer.append(", ");
				
			} else {
				first = false;
			}
			buffer.append(String.format("\"%s\"", property.toString()));
		}
		buffer.append("}");
		return buffer.toString();
	}
	
	private static String getPlotRangeOfCombinedPlot(TreeSet<GradientDescentSummary> originalRecords, TreeSet<GradientDescentSummary> adaptedLrRecords, TreeSet<GradientDescentSummary> gradientMagRecords,GraphableProperty[] axes) {
		/*
		HashMap<GraphableProperty, MaxAndMin> maxAndMinValueMap = new HashMap<>();
		for (GraphableProperty property : axes) {
			maxAndMinValueMap.put(property, findMaxAndMinValues(originalRecords, adaptedLrRecords, property));
		}
		
		StringBuffer buffer = new StringBuffer();
		buffer.append("PlotRange->{All, {0,1}}");
		
		boolean first = true;

		/*
		for (GraphableProperty property : axes) {
			if (!first) {
				buffer.append(", ");
			}
			first = false;
			if (axesType == AxesType.ExactMinAndMax) {
				buffer.append(String.format("{%f, %f}", maxAndMinValueMap.get(property).min, maxAndMinValueMap.get(property).max));
			} else {
				double extraSpace = 0;//(maxAndMinValueMap.get(property).max - maxAndMinValueMap.get(property).min) / 5;
				buffer.append(String.format("{%f, %f}", maxAndMinValueMap.get(property).min - extraSpace, maxAndMinValueMap.get(property).max + extraSpace));
			}
		}
		
		buffer.append("}");
	
		return buffer.toString();
			*/
		return "PlotRange->All";//{All, All}";
	}
	
	private static String getTicks(GraphableProperty[] axes) {
		StringBuilder[] axesTicks = new StringBuilder[axes.length];
		for (int i = 0; i < axes.length; i++) {
			GraphableProperty property = axes[i];
			axesTicks[i] = new StringBuilder();
			switch(property) {
				case GradientStoppingIteration:
				case TrainingStoppingIteration:
				case ValidationStoppingIteration:
				case ValidationStopTestError:
				case ValidationStopTrainingError:
				case ValidationStopValidationError:
				case gradientStopTestError:
				case gradientStopTrainingError:
				case gradientStopValidationError:
				case TimeInSeconds:
					axesTicks[i].append("Automatic");
					break;
				case Lambda:
					axesTicks[i].append(convertDoubleArrayToCommaSeparatedTicksList(Main.lambdas));
					break;
				case LearningRate:
					axesTicks[i].append(convertDoubleArrayToCommaSeparatedTicksList(Main.learningRates));
					break;
			}
		}
		
		// {Left, Right}, {Bottom, Top}
		StringBuilder builder = new StringBuilder();
		if (axesTicks.length == 2) {
			builder.append("FrameTicks->{ {" + axesTicks[1].toString() + ", None}, {" + axesTicks[0].toString() + ", None}}");
		} else {
			builder.append("FrameTicks->{ {" + axesTicks[1].toString() + ", None}, {" + axesTicks[0].toString() + ", None}}, {" + axesTicks[2].toString() + ", None}}");
		}

		return builder.toString();
	}
	
	public static String convertGraphablePropertyAxesArrayToMinimalString(GraphableProperty[] array) {
		StringBuffer buffer = new StringBuffer();
		boolean first = true;
		for (GraphableProperty val : array) {
			if (!first) {
				buffer.append("vs");
			}
			buffer.append(val.name());
			first = false;
		}
		return buffer.toString();
	}
	
	private static String convertGraphablePropertyAxesArrayToNiceString(GraphableProperty[] array) {
		StringBuffer buffer = new StringBuffer();
		boolean first = true;
		for (GraphableProperty val : array) {
			if (!first) {
				buffer.append(" vs. ");
			}
			buffer.append(val.toString());
			first = false;
		}
		return buffer.toString();
	}

	private static String convertDoubleArrayToCommaSeparatedTicksList(double[] array) {
		StringBuffer buffer = new StringBuffer();
		boolean first = true;
		buffer.append("{");
		double avgDist = (array[0] + array[array.length-1]) / (array.length * 4);
		double lastPrinted = array[0];
		buffer.append((String.format("%f", array[0])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1"));
		for (int i = 1; i < array.length; i++) { //double val : array) {
			// Trying to avoid collision
			if (lastPrinted + avgDist < array[i]) {
				buffer.append(", ");
				buffer.append((String.format("%f", array[i])).replaceFirst("\\.0*$|(\\.\\d*?)0+$", "$1"));
			}
		}
		buffer.append("}");
		return buffer.toString();
	}
	
	private static String convertIntArrayToCommaSeparatedTicksList(int[] array) {
		StringBuffer buffer = new StringBuffer();
		boolean first = true;
		buffer.append("{");
		double avgDist = (array[0] + array[array.length-1]) / (array.length * 4);
		double lastPrinted = Double.MIN_VALUE;
		for (int val : array) {
			// Trying to avoid collision
			if (lastPrinted + avgDist < val) {
				if (!first) {
					buffer.append(", ");
				}
				first = false;
				buffer.append(String.format("%d", val));
			}
		}
		buffer.append("}");
		return buffer.toString();
	}

	
	private static String convertDoubleArrayToCommaSeparatedList(double[] array) {
		StringBuffer buffer = new StringBuffer();
		boolean first = true;
		buffer.append("{");
		for (double val : array) {
			if (!first) {
				buffer.append(", ");
			}
			buffer.append(String.format("%f", val));
			first = false;
		}
		buffer.append("}");
		return buffer.toString();
	}
	

	
	private static MaxAndMin findMaxAndMinValues(List<GradientDescentSummary> originalRecords, List<GradientDescentSummary> adaptedLrRecords, GraphableProperty property) {
		MaxAndMin maxAndMin = new MaxAndMin();
		
		for (GradientDescentSummary record : originalRecords) {
			double value = getPropertyValue(record, property);
			if (value < maxAndMin.min) {
				maxAndMin.min = value;
			}
			if (value > maxAndMin.max) {
				maxAndMin.max = value;
			}
		}
		for (GradientDescentSummary record : adaptedLrRecords) {
			double value = getPropertyValue(record, property);
			if (value < maxAndMin.min) {
				maxAndMin.min = value;
			}
			if (value > maxAndMin.max) {
				maxAndMin.max = value;
			}
		}
		return maxAndMin;
	}
	
	private static MaxAndMin findMaxAndMinValues(List<GradientDescentSummary> records, GraphableProperty property) {
		MaxAndMin maxAndMin = new MaxAndMin();
		
		for (GradientDescentSummary record : records) {
			double value = getPropertyValue(record, property);
			if (value < maxAndMin.min) {
				maxAndMin.min = value;
			}
			if (value > maxAndMin.max) {
				maxAndMin.max = value;
			}
		}
		return maxAndMin;
	}
	
	private static double getPropertyValue(GradientDescentSummary record, GraphableProperty property) {
		switch(property) {
			case GradientStoppingIteration:
				return record.gradientStoppingIteration;
			case TrainingStoppingIteration:
				return record.trainingStoppingIteration;
			case ValidationStoppingIteration:
				return record.validationStoppingIteration;
				
			case Lambda:
				return record.parameters.lambda;
			case LearningRate:
				return record.parameters.learningRate;

			case TimeInSeconds:
				return record.timeInSeconds;
			case ValidationStopTestError:
				return record.validationStoppingTestError;
			case ValidationStopTrainingError:
				return record.validationStoppingTrainingError;
			case ValidationStopValidationError:
				return record.validationStoppingValidationError;
			case gradientStopTestError:
				return record.gradientStoppingTestError;
			case gradientStopTrainingError:
				return record.gradientStoppingTrainingError;
			case gradientStopValidationError:
				return record.gradientStoppingValidationError;
		}
		throw new IllegalArgumentException();
	}
	
	private static class UniqueXYPointKey {
		public double[] XYvalues;
		public UniqueXYPointKey(GradientDescentSummary record, GraphableProperty[] axes) {
			XYvalues = new double[axes.length-1];
			
			for (int i = 0; i < XYvalues.length; i++) {
				XYvalues[i] = getPropertyValue(record, axes[i]);
			}
		}
		
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + Arrays.hashCode(XYvalues);
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			UniqueXYPointKey other = (UniqueXYPointKey) obj;
			if (!Arrays.equals(XYvalues, other.XYvalues))
				return false;
			return true;
		}
	}
	
	private static class Point implements Comparable<Point> {
		private double[] values;
		
		public Point(GradientDescentSummary record, GraphableProperty[] axes) {
			values = new double[axes.length];
			for (int i = 0; i < axes.length; i++) {
				values[i] = getPropertyValue(record, axes[i]);
			}
		}
		
		public Point(UniqueXYPointKey key, SumCountAverage averageZValue) {
			values = new double[key.XYvalues.length + 1];
			for (int i = 0; i < key.XYvalues.length; i++) {
				values[i] = key.XYvalues[i];
			}
			values[key.XYvalues.length] = averageZValue.getMean();
		}
		
		public String getMathematicaListEntry() {
			return convertDoubleArrayToCommaSeparatedList(values);
		}

		@Override
		public int compareTo(Point that) {
			for (int i = 0; i < this.values.length; i++) {
				if (this.values[i] < that.values[i]) {
					return -1;
				}
				if (this.values[i] > that.values[i]) {
					return 1;
				}
			}
			return 0;
		}
	}
}
