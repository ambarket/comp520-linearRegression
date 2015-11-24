package linearRegressor;
import java.io.BufferedWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import dataset.Dataset;
import dataset.DatasetParameters;
import dataset.LinearRegressorDataset;
import utilities.DoubleCompare;
import utilities.MathematicaListCreator;
import utilities.RandomSample;
import utilities.StopWatch;

public class Main {
	public static String RESULTS_DIRECTORY = System.getProperty("user.dir") + "/results/";
	
	private static int NUMBER_OF_RUNS = 10;
	public static double TRAINING_SAMPLE_FRACTION = 0.8;
	public static DatasetParameters powerPlantParameters = new DatasetParameters("powerPlant", "Power Plant", "data/PowerPlant/", "Folds5x2_pp.txt",4);
	public static DatasetParameters nasaParameters = new DatasetParameters("nasa", "Nasa Air Foil", "data/NASAAirFoild/", "data.txt",5);
	public static DatasetParameters bikeSharingDayParameters = new DatasetParameters("bikeSharingDay", "Bike Sharing By Day", "data/BikeSharing/", "bikeSharing.txt",11);
	public static DatasetParameters crimeCommunitiesParameters = new DatasetParameters("crimeCommunities", "Crime Communities", "data/CrimeCommunities/", "communitiesOnlyPredictive.txt",122);
	
	public static DatasetParameters[] datasets = new DatasetParameters[] {/*nasaParameters, */powerPlantParameters /*, crimeCommunitiesParameters*/};
	public static UpdateRule[] updateRules = new UpdateRule[] {UpdateRule.Original, UpdateRule.AdaptedLR};
	public static double[] learningRates = new double[] {/*0.0001, 0.001, 0.01, 0.1, .5, 1*/ 0.5};
	public static double[] lambdas = new double[] {0.0, 0.05, 0.1, 0.5, 1, 5};
	public static int maxNumberOfIterations = 1000000;
	
	public static ExecutorService executorService = Executors.newFixedThreadPool(3);
	//public static ExecutorService executorService = Executors.newCachedThreadPool();
	public static Queue<Future<Void>> futureQueue = new LinkedList<Future<Void>>();
	
	public static void main(String[] args) {
		//generateLearningCurveForDerivativeSolvers();
		//generateRunData();
		generateGradientDescentRunData();
	}
	
	public static void generateRunData() {
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
				runTimer.start();
				LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
				
				for (int numberOfExamples = unpartitionedDataset.numberOfPredictorsPlus1+1; 
						numberOfExamples < unpartitionedDataset.numberOfAllTrainingExamples-1; 
						numberOfExamples++) 
				{
					LinearRegressor lr = new LinearRegressor( new LinearRegressorDataset(unpartitionedDataset, numberOfExamples), runNumber);

					futureQueue.add(executorService.submit(new DerivativeSolverTask(lr, numberOfExamples, runNumber)));

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
	
	final static double VALIDATION_SAMPLE_FRACTION = 0.25;
	public static void generateGradientDescentRunData() {
		
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
				runTimer.start();
				LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
				int tmp = (int)(unpartitionedDataset.numberOfAllTrainingExamples * (1 - VALIDATION_SAMPLE_FRACTION));
				LinearRegressor lr = new LinearRegressor( new LinearRegressorDataset(unpartitionedDataset, tmp), runNumber); // train without validation set
				
				for (UpdateRule updateRule : updateRules) {
					for (double learningRate : learningRates) {
						for (double lambda : lambdas) {
							futureQueue.add(executorService.submit(
									new GradientDescentTask(lr,
											new GradientDescentParameters(runNumber, 
													"StandardValidation",
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
				
				while (futureQueue.size() > 4) {
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
	final static int NUMBER_OF_FOLDS = 5;
	public static void generateGradientDescentRunDataCrossValidation() {
		
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
				runTimer.start();
				LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
				LinearRegressor[] foldsPlusUnpartitionedModels = new LinearRegressor[NUMBER_OF_FOLDS+1];
				foldsPlusUnpartitionedModels[NUMBER_OF_FOLDS] = new LinearRegressor( unpartitionedDataset, runNumber); // train without validation set
				
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
					foldsPlusUnpartitionedModels[i] = new LinearRegressor(new LinearRegressorDataset(unpartitionedDataset, trainingInEachFold[i], numberOfTrainingExamplesInEachModel), runNumber);
				}

				for (UpdateRule updateRule : updateRules) {
					for (double learningRate : learningRates) {
						for (double lambda : lambdas) {
							for (int i = 0; i < NUMBER_OF_FOLDS+1; i++) {
								String foldId = (i < NUMBER_OF_FOLDS) ? "Fold" + i : "AllTrainingData";
								futureQueue.add(executorService.submit(
										new GradientDescentTask(foldsPlusUnpartitionedModels[i],
												new GradientDescentParameters(runNumber, 
														foldId,
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
	
	public static void generateLearningCurveForDerivativeSolvers() {
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, TRAINING_SAMPLE_FRACTION));
			DerivateSolverResult[] averageResults = new DerivateSolverResult[unpartitionedDataset.numberOfAllTrainingExamples];
			
			
			for (int numberOfExamples = unpartitionedDataset.numberOfPredictorsPlus1+1; 
					numberOfExamples < unpartitionedDataset.numberOfAllTrainingExamples-1; 
					numberOfExamples++) 
			{
				for (int runNumber = 0; runNumber < 1; runNumber++) {
					runTimer.start();
					DerivateSolverResult result = DerivativeSolverTask.readOptimalWeightsBySolvingDerivate(dsParam, numberOfExamples, runNumber);
					//futureQueue.add(executorService.submit(new DerivativeSolverTask(lr, numberOfExamples, runNumber)));
					
					if (averageResults[numberOfExamples] == null) {
						averageResults[numberOfExamples] = result;
					} else {
						averageResults[numberOfExamples].trainingError += result.trainingError;
						averageResults[numberOfExamples].validationError += result.validationError;
						averageResults[numberOfExamples].testError += result.testError;
					}
				}
			}
			
			ArrayList<Double> trainingError = new ArrayList<Double>();
			ArrayList<Double> validationError = new ArrayList<Double>();
			ArrayList<Double> testError = new ArrayList<Double>();
			int optimalValidationErrorIteration = 0;
			double optimalValidationError = Double.MAX_VALUE;
			double maxRMSE = Double.MIN_VALUE;
			for (DerivateSolverResult result : averageResults) {
				if (result != null) {
					trainingError.add(result.trainingError / NUMBER_OF_RUNS);
					validationError.add(result.validationError / NUMBER_OF_RUNS);
					testError.add(result.testError / NUMBER_OF_RUNS);
					if (DoubleCompare.lessThan(validationError.get(validationError.size()-1), optimalValidationError)) {
						optimalValidationError = validationError.get(validationError.size()-1);
						optimalValidationErrorIteration = validationError.size()-1;
					}
					if (DoubleCompare.greaterThan(trainingError.get(validationError.size()-1), maxRMSE)) {
						maxRMSE = trainingError.get(trainingError.size()-1);
					}
					if (DoubleCompare.greaterThan(validationError.get(validationError.size()-1), maxRMSE)) {
						maxRMSE = validationError.get(validationError.size()-1);
					}
					if (DoubleCompare.greaterThan(testError.get(testError.size()-1), maxRMSE)) {
						maxRMSE = testError.get(testError.size()-1);
					}
				}
			}
			String trainingErrorList = MathematicaListCreator.convertToMathematicaList(trainingError, unpartitionedDataset.numberOfPredictorsPlus1+1);
			String validationErrorList = MathematicaListCreator.convertToMathematicaList(validationError, unpartitionedDataset.numberOfPredictorsPlus1+1);
			String testErrorList = MathematicaListCreator.convertToMathematicaList(testError, unpartitionedDataset.numberOfPredictorsPlus1+1);
			optimalValidationErrorIteration += unpartitionedDataset.numberOfPredictorsPlus1+1;
			String optimalValidationErrorList = "{{" + optimalValidationErrorIteration + ", 0}, {" + optimalValidationErrorIteration + ", " + maxRMSE + "}}";
			
			String directory = String.format("%s/%s/", Main.RESULTS_DIRECTORY, dsParam.minimalName);
			
			String imageFileNameNoExtension = "derivateSetToZeroLearningCurve";
			try {
				BufferedWriter bw = new BufferedWriter(new PrintWriter(directory + imageFileNameNoExtension + ".txt"));
				bw.write("trainingError := " + trainingErrorList + "\n");
				bw.write("validationError := " + validationErrorList + "\n");
				bw.write("testError := " + testErrorList + "\n");
				bw.write("optimalNumberOfExamples := " + optimalValidationErrorList + "\n");
				bw.write("learningCurve := ListLinePlot[{trainingError, validationError, testError, optimalNumberOfExamples}"
						+ ", PlotLegends -> {\"trainingError\", \"validationError\", \"testError\", \"optimalNumberOfExamples\"}"
						+ ", PlotStyle -> {{Magenta, Dashed}, Blue, Red, {Green, Thin}}"
						+ ", AxesLabel->{\"Number Of Training Examples\", \"RMSE\"}"
						+ ", PlotRange -> {{Automatic, Automatic}, {0, " + (.25 * maxRMSE) + "}}"
						+ "] \nlearningCurve\n\n");
				
				StringBuffer saveToFile = new StringBuffer();
				StringBuffer latexCode = new StringBuffer();
				
				
				saveToFile.append("fileName := \"" + imageFileNameNoExtension + "\"\n");
				saveToFile.append("Export[fileName <> \".png\", learningCurve]\n\n");

				latexCode.append("\\begin{figure}[!htb]\\centering\n");
				latexCode.append("\\includegraphics[width=1\\textwidth]{{" + imageFileNameNoExtension + "}.png}\n");
				latexCode.append("\\caption{" + dsParam.fileName + " " + imageFileNameNoExtension + "}\n");
				latexCode.append("\\label{fig:" +  dsParam.minimalName + imageFileNameNoExtension + "}\n");
				latexCode.append("\\end{figure}\n\n");
				
				bw.write(saveToFile.toString() + "\n");
				bw.write(latexCode.toString() + "\n");
				bw.close();
			} catch (Exception e) {
				
			}
		}
	}	
}
