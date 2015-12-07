package linearRegressor;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import dataset.Dataset;
import dataset.DatasetParameters;
import dataset.LinearRegressorDataset;
import utilities.CommandLineExecutor;
import utilities.MathematicaListCreator;
import utilities.RandomSample;
import utilities.StopWatch;

public class Main {
	public static String RESULTS_DIRECTORY = "/mnt/nfs/Austin/comp520-linearRegression/results/";
	public static String LOCKS_DIRECTORY = "/mnt/nfs/Austin/comp520-linearRegression/locks/";
	private static int NUMBER_OF_RUNS = 10;
	public static double TRAINING_SAMPLE_FRACTION = 0.8;
	public static DatasetParameters powerPlantParameters = new DatasetParameters("powerPlant", "Power Plant", "data/PowerPlant/", "Folds5x2_pp.txt",4);
	public static DatasetParameters nasaParameters = new DatasetParameters("nasa", "Nasa Air Foil", "data/NASAAirFoild/", "data.txt",5);
	public static DatasetParameters bikeSharingDayParameters = new DatasetParameters("bikeSharingDay", "Bike Sharing By Day", "data/BikeSharing/", "bikeSharing.txt",11);
	public static DatasetParameters crimeCommunitiesParameters = new DatasetParameters("crimeCommunities", "Crime Communities", "data/CrimeCommunities/", "communitiesOnlyPredictive.txt",122);
	public static DatasetParameters abdominalCircumference = new DatasetParameters("ac", "Abdominal Circumference", "data/AbdominalCircumference/", "ac.txt",1);
	public static DatasetParameters simpleLine = new DatasetParameters("line", "Simple Line", "data/AbdominalCircumference/", "line.txt",1);
	public static DatasetParameters[] datasets = new DatasetParameters[] {/*simpleLine, /*abdominalCircumference, nasaParameters,*/ powerPlantParameters /*, crimeCommunitiesParameters*/};
	public static UpdateRule[] updateRules = new UpdateRule[] {UpdateRule.GradientMag, UpdateRule.Original, UpdateRule.AdaptedLR};
	public static double[] learningRates = new double[] {0.0001, 0.001, 0.01, 0.1, .5, 1};
	public static double[] lambdas = new double[] {0.0, 0.05, 0.1, 0.5, 1, 5};
	public static int maxNumberOfIterations = 500000;
	
	public static ExecutorService executorService = Executors.newFixedThreadPool(12);
	//public static ExecutorService executorService = Executors.newCachedThreadPool();
	public static Queue<Future<Void>> futureQueue = new LinkedList<Future<Void>>();
	
	public static void main(String[] args) {
		LatexResultsGenerator.writeEntireResultsSection(powerPlantParameters);
		
		generateDerivativeSolverRunData();
		generateLearningCurveForDerivativeSolvers();

		generateGradientDescentRunData();
		averageGradientDescentRunData();
		
		gradientDescentLearningCurvesForBestParameters();
		
		GradientDescentSummary.readSortAndSaveGradientDescentSummaries();
		GradientDescentSummaryGraphGenerator.generateAndSaveGraphs(powerPlantParameters);
		
		GradientDescentInformation.generateAndExecutePlotLegend(powerPlantParameters);
		GradientDescentSummaryGraphGenerator.generateAndExecutePlotLegend(powerPlantParameters);
		generateAndExecuteLearningCurveLegend(powerPlantParameters);
		
		LatexResultsGenerator.writeEntireResultsSection(powerPlantParameters);
		executorService.shutdownNow();
	}

	
	public static void gradientDescentLearningCurvesForBestParameters() {
		GradientDescentSummary bestValOriginal = null, bestValAdaptedLR = null, bestValGradientMag = null,
				bestGradOriginal = null, bestGradAdaptedLR = null, bestGradGradientMag = null;
		TreeSet<GradientDescentSummary> all = GradientDescentSummary.readGradientDescentSummaries(powerPlantParameters);
		
		TreeSet<GradientDescentSummary> validationStopRecords = new TreeSet<>(new GradientDescentSummary.ValidationStopComparator());
		validationStopRecords.addAll(all);
		
		TreeSet<GradientDescentSummary> gradientStopRecords = new TreeSet<>(new GradientDescentSummary.GradientStopComparator());
		gradientStopRecords.addAll(all);
		
		for (GradientDescentSummary summary : validationStopRecords) {
			if (bestValOriginal != null && bestValAdaptedLR != null && bestValGradientMag != null) {
				break;
			}
			switch (summary.parameters.updateRule) {
				case AdaptedLR:
					if (bestValAdaptedLR == null) {
						bestValAdaptedLR = summary;
					}
					break;
				case GradientMag:
					if (bestValGradientMag == null) {
						bestValGradientMag = summary;
					}
					break;
				case Original:
					if (bestValOriginal == null) {
						bestValOriginal = summary;
					}
					break;
				default:
					break;
				
			}
		}
		
		for (GradientDescentSummary summary : gradientStopRecords) {
			if (bestGradOriginal != null && bestGradAdaptedLR != null && bestGradGradientMag != null) {
				break;
			}
			switch (summary.parameters.updateRule) {
				case AdaptedLR:
					if (bestGradAdaptedLR == null) {
						bestGradAdaptedLR = summary;
					}
					break;
				case GradientMag:
					if (bestGradGradientMag == null) {
						bestGradGradientMag = summary;
					}
					break;
				case Original:
					if (bestGradOriginal == null) {
						bestGradOriginal = summary;
					}
					break;
				default:
					break;
				
			}
		}
		
		generateAndExecuteLearningCurveLegend(powerPlantParameters);
		generateGradientDescentLearningCurveData(powerPlantParameters, UpdateRule.Original, bestValOriginal.parameters.learningRate, bestValOriginal.parameters.lambda);
		generateLearningCurveForGradientDescent(powerPlantParameters, UpdateRule.Original, bestValOriginal.parameters.learningRate, bestValOriginal.parameters.lambda);
		
		generateGradientDescentLearningCurveData(powerPlantParameters, UpdateRule.AdaptedLR, bestValAdaptedLR.parameters.learningRate, bestValAdaptedLR.parameters.lambda);
		generateLearningCurveForGradientDescent(powerPlantParameters, UpdateRule.AdaptedLR, bestValAdaptedLR.parameters.learningRate, bestValAdaptedLR.parameters.lambda);
		
		generateGradientDescentLearningCurveData(powerPlantParameters, UpdateRule.GradientMag, bestValGradientMag.parameters.learningRate, bestValGradientMag.parameters.lambda);
		generateLearningCurveForGradientDescent(powerPlantParameters, UpdateRule.GradientMag, bestValGradientMag.parameters.learningRate, bestValGradientMag.parameters.lambda);
		
		
		generateAndExecuteLearningCurveLegend(powerPlantParameters);
		generateGradientDescentLearningCurveData(powerPlantParameters, UpdateRule.Original, bestGradOriginal.parameters.learningRate, bestGradOriginal.parameters.lambda);
		generateLearningCurveForGradientDescent(powerPlantParameters, UpdateRule.Original, bestGradOriginal.parameters.learningRate, bestGradOriginal.parameters.lambda);
		
		generateGradientDescentLearningCurveData(powerPlantParameters, UpdateRule.AdaptedLR, bestGradAdaptedLR.parameters.learningRate, bestGradAdaptedLR.parameters.lambda);
		generateLearningCurveForGradientDescent(powerPlantParameters, UpdateRule.AdaptedLR, bestGradAdaptedLR.parameters.learningRate, bestGradAdaptedLR.parameters.lambda);
		
		generateGradientDescentLearningCurveData(powerPlantParameters, UpdateRule.GradientMag, bestGradGradientMag.parameters.learningRate, bestGradGradientMag.parameters.lambda);
		generateLearningCurveForGradientDescent(powerPlantParameters, UpdateRule.GradientMag, bestGradGradientMag.parameters.learningRate, bestGradGradientMag.parameters.lambda);
	}

	
	final static double VALIDATION_SAMPLE_FRACTION = 0.25;
	public static void generateGradientDescentRunData() {
		
		int total = updateRules.length * learningRates.length * lambdas.length;
		
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
				int done = 0;
				runTimer.start();
				LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
				int tmp = (int)(unpartitionedDataset.numberOfAllTrainingExamples * (1 - VALIDATION_SAMPLE_FRACTION));
				LinearRegressor lr = new LinearRegressor( new LinearRegressorDataset(unpartitionedDataset, tmp));
				
				String resultsSubDirectory = String.format("%s/gradientDescent/Run%d/%s/", 
						dsParam.minimalName,
						runNumber,
						"StandardValidation"
					);
				
				for (UpdateRule updateRule : updateRules) {
					for (double learningRate : learningRates) {
						for (double lambda : lambdas) {
							futureQueue.add(executorService.submit(
									new GradientDescentTask(lr, runNumber, ++done, total,
											new GradientDescentParameters(resultsSubDirectory,
													lr.dataset, 
													maxNumberOfIterations, 
													updateRule, 
													learningRate, 
													lambda)
											)
									)
								);
						}
					}
				}
				
				while (futureQueue.size() > 15) {
					try {
						futureQueue.poll().get();
					} catch (InterruptedException | ExecutionException e) {
						e.printStackTrace();
						System.exit(1);
					} 
				}
							
				runTimer.printMessageWithTime("Finished run " + runNumber);
			}
			while (!futureQueue.isEmpty()) {
				try {
					futureQueue.poll().get();
				} catch (InterruptedException | ExecutionException e) {
					e.printStackTrace();
					System.exit(1);
				} 
			}
		}
	}
	
	public static void averageGradientDescentRunData() {
		for (DatasetParameters dsParam : datasets) {
			LinearRegressorDataset ds = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
			int tmp = (int)(ds.numberOfAllTrainingExamples * (1 - VALIDATION_SAMPLE_FRACTION));
			ds = new LinearRegressorDataset(ds, tmp);
			
			int done = 0;
			int total = updateRules.length * learningRates.length * lambdas.length;
			Queue<Future<Void>> futureQueue = new LinkedList<>();
			for (UpdateRule updateRule : updateRules) {
				for (double learningRate : learningRates) {
					for (double lambda : lambdas) {
						String resultsSubDirectory = String.format("%s/gradientDescent/Averages/%s/", 
								dsParam.minimalName,
								"StandardValidation"
							);
						GradientDescentParameters avg = new GradientDescentParameters(resultsSubDirectory,
								ds, 
								maxNumberOfIterations, 
								updateRule, 
								learningRate, 
								lambda);
						futureQueue.add(executorService.submit(new AverageGradientDescentRunDataTask(ds, NUMBER_OF_RUNS, ++done, total, avg)));
						while (futureQueue.size() > 15) {
							try {
								futureQueue.poll().get();
							} catch (InterruptedException | ExecutionException e) {
								e.printStackTrace();
								System.exit(1);
							} 
						}
					}
					while (!futureQueue.isEmpty()) {
						try {
							futureQueue.poll().get();
						} catch (InterruptedException | ExecutionException e) {
							e.printStackTrace();
							System.exit(1);
						} 
					}
				}
			}
		}
	}
	
	

	final static int NUMBER_OF_FOLDS = 5;
	public static void generateGradientDescentRunDataCrossValidation() {
		
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
				runTimer.start();
				LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
				LinearRegressor[] foldsPlusUnpartitionedModels = new LinearRegressor[NUMBER_OF_FOLDS+1];
				foldsPlusUnpartitionedModels[NUMBER_OF_FOLDS] = new LinearRegressor( unpartitionedDataset); // train without validation set
				
				// Partition the data set into k folds. All done with boolean index masks into the original dataset
				int[] shuffledIndices = (new RandomSample()).fisherYatesShuffle(unpartitionedDataset.numberOfAllTrainingExamples);
				int foldSize = shuffledIndices.length / NUMBER_OF_FOLDS;
				boolean[][] trainingInEachFold = new boolean[NUMBER_OF_FOLDS+1][shuffledIndices.length];
				int numberOfTrainingExamplesInEachModel = ((NUMBER_OF_FOLDS-1)*foldSize);
				for (int i = 0; i < NUMBER_OF_FOLDS; i++) {
					int first = i * foldSize, last = (i * foldSize) + (numberOfTrainingExamplesInEachModel);
					for (int j = first; j < last; j++) {
						int safeIndex = j % shuffledIndices.length;
						trainingInEachFold[i][shuffledIndices[safeIndex]] = true;
					}
					foldsPlusUnpartitionedModels[i] = new LinearRegressor(new LinearRegressorDataset(unpartitionedDataset, trainingInEachFold[i], numberOfTrainingExamplesInEachModel));
				}

				for (UpdateRule updateRule : updateRules) {
					for (double learningRate : learningRates) {
						for (double lambda : lambdas) {
							for (int i = 0; i < NUMBER_OF_FOLDS+1; i++) {
								String foldId = (i < NUMBER_OF_FOLDS) ? "Fold" + i : "AllTrainingData";
								
								String resultsSubDirectory = String.format("%s/gradientDescent/Run%d/%s/", 
										dsParam.minimalName,
										runNumber,
										foldId
									);
								
								futureQueue.add(executorService.submit(
										new GradientDescentTask(foldsPlusUnpartitionedModels[i], runNumber, 0, 0,
												new GradientDescentParameters(resultsSubDirectory,
														foldsPlusUnpartitionedModels[i].dataset, 
														maxNumberOfIterations, 
														updateRule, 
														learningRate, 
														lambda)
												)
										)
									);
							}
						}
					}
				}
				
				if (futureQueue.size() > 8) {
					while (futureQueue.size() > 4) {
						try {
							futureQueue.poll().get();
						} catch (InterruptedException e) {
							e.printStackTrace();
						} catch (ExecutionException e) {
							e.printStackTrace();
						}
					}
				}
							
				runTimer.printMessageWithTime("Finished run " + runNumber);
			}
			while (!futureQueue.isEmpty()) {
				try {
					futureQueue.poll().get();
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	public static void generateGradientDescentLearningCurveData(DatasetParameters dsParam, UpdateRule updateRule, double learningRate, double lambda) {
		StopWatch runTimer = new StopWatch();
		runTimer.start();
		LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));

		String resultsSubDirectory = String.format("%s/gradientDescentLearningCurve/", 
				dsParam.minimalName
			);

		for (int numberOfExamples = unpartitionedDataset.numberOfPredictorsPlus1+1; 
				numberOfExamples < 200 + unpartitionedDataset.numberOfPredictorsPlus1+1;//unpartitionedDataset.numberOfAllTrainingExamples-1; 
				numberOfExamples++) 
		{
			LinearRegressor lr = new LinearRegressor( new LinearRegressorDataset(unpartitionedDataset, numberOfExamples));

			futureQueue.add(executorService.submit(
					new GradientDescentLearningCurveTask(lr, 0, numberOfExamples, unpartitionedDataset.numberOfAllTrainingExamples-1,
							new GradientDescentParameters(resultsSubDirectory, 
									lr.dataset, 
									maxNumberOfIterations, 
									updateRule, 
									learningRate, 
									lambda)
							)
					)
				);

			if (futureQueue.size() > 12) {
				while (futureQueue.size() > 6) {
					try {
						futureQueue.poll().get();
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}
			}
		}				
		runTimer.printMessageWithTime("Finished run " + 0);
		while (!futureQueue.isEmpty()) {
			try {
				futureQueue.poll().get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static void generateLearningCurveForGradientDescent(DatasetParameters dsParam, UpdateRule updateRule, double learningRate, double lambda) {
		StopWatch runTimer = new StopWatch();
		runTimer.start();
		LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));

		String resultsSubDirectory = String.format("%s/gradientDescentLearningCurve/", 
				dsParam.minimalName
			);
		GradientDescentParameters generalParams = new GradientDescentParameters(resultsSubDirectory,
				unpartitionedDataset, 
				maxNumberOfIterations, 
				updateRule, 
				learningRate, 
				lambda);
		
		ArrayList<LearningCurveEntry> entries = new ArrayList<>();
		for (int numberOfExamples = unpartitionedDataset.numberOfPredictorsPlus1+1; 
				numberOfExamples < 200 + unpartitionedDataset.numberOfPredictorsPlus1+1;//unpartitionedDataset.numberOfAllTrainingExamples-1; 
				numberOfExamples++) 
		{
			LinearRegressorDataset dataset = new LinearRegressorDataset(unpartitionedDataset, numberOfExamples);

			entries.add(GradientDescentLearningCurveTask.readLearningCurveFile(new GradientDescentParameters(resultsSubDirectory,
					dataset, 
					maxNumberOfIterations, 
					updateRule, 
					learningRate, 
					lambda)));			
		}				
		runTimer.printMessageWithTime("Finished reading gradient descent learning curve entries " + 0);
		
		generateGenericLearningCurve(dsParam, entries, "gradientDesenctLearningCurve", "gradientDescentLearningCurve-" + generalParams.fileNamePrefix);
	}
	
	public static void generateAndExecuteLearningCurveLegend(DatasetParameters dsParam) {
		String file = Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescentLearningCurve/learningCurveLegend";
		
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(file + ".m"));
			bw.append("learningCurveLegend = LineLegend[{Red, Cyan, Black}, {\"Training Error\", \"Validation Error\", \"Test Error\"}]\n\n");
			bw.append("fileName = \"" + file  + "\"\n");
			bw.append("Export[fileName <> \".png\", learningCurveLegend, ImageResolution -> 300]\n\n");
			bw.flush();
			bw.close();
			CommandLineExecutor.executeMathematicaScript(Main.RESULTS_DIRECTORY + dsParam.minimalName + "/gradientDescentLearningCurve/", "learningCurveLegend.m");
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

	}
	
	public static void generateGenericLearningCurve(DatasetParameters dsParam, List<LearningCurveEntry> entries, String subDirectory, String imageFileNameNoExtension) {
		ArrayList<Double> trainingError = new ArrayList<Double>();
		ArrayList<Double> validationError = new ArrayList<Double>();
		ArrayList<Double> testError = new ArrayList<Double>();
		int optimalValidationErrorIteration = 0;
		double optimalValidationError = Double.MAX_VALUE;
		double maxRMSE = Double.MIN_VALUE;
		for (LearningCurveEntry result : entries) {
			if (result != null) {
				trainingError.add(result.trainingError);
				validationError.add(result.validationError);
				testError.add(result.testError);
				if (validationError.get(validationError.size()-1) < optimalValidationError) {
		//			optimalValidationError = validationError.get(validationError.size()-1);
		//			optimalValidationErrorIteration = validationError.size()-1;
				}
				if (trainingError.get(validationError.size()-1) > maxRMSE) {
					maxRMSE = trainingError.get(trainingError.size()-1);
				}
				if (validationError.get(validationError.size()-1) > maxRMSE) {
					maxRMSE = validationError.get(validationError.size()-1);
				}
				if (testError.get(testError.size()-1) > maxRMSE) {
					maxRMSE = testError.get(testError.size()-1);
				}
			}
		}
		int firstEntryXValue = entries.get(0).numberOfTrainingExamples;
		String trainingErrorList = MathematicaListCreator.convertToMathematicaList(trainingError, firstEntryXValue);
		String validationErrorList = MathematicaListCreator.convertToMathematicaList(validationError, firstEntryXValue);
		String testErrorList = MathematicaListCreator.convertToMathematicaList(testError, firstEntryXValue);
		//optimalValidationErrorIteration += firstEntryXValue;
		//String optimalValidationErrorList = "{{" + optimalValidationErrorIteration + ", 0}, {" + optimalValidationErrorIteration + ", " + maxRMSE + "}}";
		
		String directory = String.format("%s/%s/%s/", Main.RESULTS_DIRECTORY, dsParam.minimalName, subDirectory);
		new File(directory).mkdirs();
		
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(directory + imageFileNameNoExtension + ".m"));
			bw.write("trainingError := " + trainingErrorList + "\n");
			bw.write("validationError := " + validationErrorList + "\n");
			bw.write("testError := " + testErrorList + "\n");
			String plotRangePadding = "PlotRangePadding->{{Scaled[0.03],Scaled[0.03]}, {Scaled[0.03], Scaled[0.03]}}";
			String imageMargins = "ImageMargins->{{0,0},{5,5}}";
			bw.write((String.format("%s = %s[%s, %s, %s, %s, %s, %s]\n", 
					"learningCurve", 
					"ListLinePlot",
					("{trainingError, validationError, testError}"), 
					"PlotRange -> {{Automatic, Automatic}, {0, " + "Automatic" + "}}", 
					"PlotStyle -> {{Red, Opacity[0.5]}, {Cyan, Opacity[0.5]}, {Black, Opacity[0.5]}}",
					//"PlotMarkers -> {Automatic, Medium}",
					"Frame->True, FrameStyle->Black , FrameTicksStyle->Black, LabelStyle->{Black, 12}, FrameLabel->{\"Number of Training Examples\" , \"RMSE\"}",
					plotRangePadding,
					imageMargins)
				));
			//bw.write("optimalNumberOfExamples := " + optimalValidationErrorList + "\n");
			
			StringBuffer saveToFile = new StringBuffer();
			StringBuffer latexCode = new StringBuffer();
			

		
			saveToFile.append("fileName := \"" + imageFileNameNoExtension + "\"\n");
			saveToFile.append("Export[fileName <> \".png\", learningCurve, ImageResolution -> 300]\n\n");

			latexCode.append("\\begin{figure}[!htb]\\centering\n");
			latexCode.append("\\includegraphics[width=1\\textwidth]{{" + imageFileNameNoExtension + "}.png}\n");
			latexCode.append("\\caption{" + dsParam.fileName + " " + imageFileNameNoExtension + "}\n");
			latexCode.append("\\label{fig:" +  dsParam.minimalName + imageFileNameNoExtension + "}\n");
			latexCode.append("\\end{figure}\n\n");
			
			bw.write(saveToFile.toString() + "\n");
			bw.write(latexCode.toString() + "\n");
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		StopWatch timer = new StopWatch().start();
		try {
			CommandLineExecutor.executeMathematicaScript(directory, imageFileNameNoExtension + ".m");
		
			timer.printMessageWithTime("Executed learning curve script");
		} catch (Exception e) {
			System.err.println(StopWatch.getDateTimeStamp());
			e.printStackTrace();
		}
	}
	
	
	public static void generateDerivativeSolverRunData() {
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
				runTimer.start();
				LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
				
				for (int numberOfExamples = unpartitionedDataset.numberOfPredictorsPlus1+1; 
						numberOfExamples < 200 + unpartitionedDataset.numberOfPredictorsPlus1+1;//unpartitionedDataset.numberOfAllTrainingExamples-1; 
						numberOfExamples++) 
				{
					LinearRegressor lr = new LinearRegressor( new LinearRegressorDataset(unpartitionedDataset, numberOfExamples));

					futureQueue.add(executorService.submit(new DerivativeSolverTask(lr, numberOfExamples, runNumber)));

					if (futureQueue.size() > 20) {
						while (futureQueue.size() > 10) {
							try {
								futureQueue.poll().get();
							} catch (InterruptedException e) {
								e.printStackTrace();
							} catch (ExecutionException e) {
								e.printStackTrace();
							}
						}
					}
					
				}				
			}
			while (!futureQueue.isEmpty()) {
				try {
					futureQueue.poll().get();
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
		}
		
	}
	
	public static void generateLearningCurveForDerivativeSolvers() {
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
			ArrayList<LearningCurveEntry> averageResults = new ArrayList<>(300 + unpartitionedDataset.numberOfPredictorsPlus1+1);//LearningCurveEntry[200 + unpartitionedDataset.numberOfPredictorsPlus1+1];
			
			
			
			for (int numberOfExamples = unpartitionedDataset.numberOfPredictorsPlus1+1; 
					numberOfExamples < 200 + unpartitionedDataset.numberOfPredictorsPlus1+1;// unpartitionedDataset.numberOfAllTrainingExamples-1; 
					numberOfExamples++) 
			{
				for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
					runTimer.start();
					LearningCurveEntry result = DerivativeSolverTask.readOptimalWeightsBySolvingDerivative(dsParam, numberOfExamples, runNumber);
					
					if (averageResults.size() == numberOfExamples-(unpartitionedDataset.numberOfPredictorsPlus1+1)) {
						averageResults.add(result);
					} else if (averageResults.size() > numberOfExamples-(unpartitionedDataset.numberOfPredictorsPlus1+1)) {
						averageResults.get(numberOfExamples-(unpartitionedDataset.numberOfPredictorsPlus1+1)).trainingError += result.trainingError;
						averageResults.get(numberOfExamples-(unpartitionedDataset.numberOfPredictorsPlus1+1)).validationError += result.validationError;
						averageResults.get(numberOfExamples-(unpartitionedDataset.numberOfPredictorsPlus1+1)).testError += result.testError;
					} else {
						throw new IllegalStateException();
					}
				}
				averageResults.get(numberOfExamples-(unpartitionedDataset.numberOfPredictorsPlus1+1)).trainingError /= NUMBER_OF_RUNS;
				averageResults.get(numberOfExamples-(unpartitionedDataset.numberOfPredictorsPlus1+1)).validationError  /= NUMBER_OF_RUNS;
				averageResults.get(numberOfExamples-(unpartitionedDataset.numberOfPredictorsPlus1+1)).testError  /= NUMBER_OF_RUNS;
			}
			String imageFileNameNoExtension = "derivateSetToZeroLearningCurve";
			generateGenericLearningCurve(dsParam, averageResults, "derivativeSolverLearningCurve", imageFileNameNoExtension);
		}
	}
}
