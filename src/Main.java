import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;

import parameterTuning.ParameterTuningParameters;
import sun.security.util.DerInputStream;
import utilities.StopWatch;
import dataset.Dataset;
import dataset.DatasetParameters;

public class Main {
	private static int NUMBER_OF_RUNS = 1;
	
	public static void main(String[] args) {
		ParameterTuningParameters.nasaParameters.fileDirectory = ParameterTuningParameters.nasaParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");
		ParameterTuningParameters.crimeCommunitiesParameters.fileDirectory = ParameterTuningParameters.crimeCommunitiesParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");
		ParameterTuningParameters.powerPlantParameters.fileDirectory = ParameterTuningParameters.powerPlantParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");
		
		DatasetParameters[] datasets = new DatasetParameters[] {ParameterTuningParameters.nasaParameters, ParameterTuningParameters.powerPlantParameters, ParameterTuningParameters.crimeCommunitiesParameters};
		UpdateRule[] updateRules = new UpdateRule[] {UpdateRule.Original, UpdateRule.AdaptedLR};
		double[] learningRates = new double[] {0.0001, 0.001, 0.01, 0.1, .5, 1};
		double[] lambdas = new double[] {0.0, 0.1, 1, 5};

		ExecutorService executorService = Executors.newFixedThreadPool(4);
		Queue<Future<Void>> futureQueue = new LinkedList<Future<Void>>();
		int maxNumberOfIterations = 1000000;
		StopWatch runTimer = new StopWatch();
		for (DatasetParameters dsParam : datasets) {
			for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
				runTimer.start();
				LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, ParameterTuningParameters.TRAINING_SAMPLE_FRACTION));
				
				/**
				 * Loop from a training set with numberOfPredictorsPlus1 examples, validation set with numberOfAllTrainingExamples examples
				 * 						set with numberOfAllTrainingExamples-1 examples, validation set with 1 example.
				 * 
				 * Training set must minimally have as many rows as it does columns in order for the inverse calculation to work. Throws rank deficient error otherwise.
				 */
				for (int numberOfExamples = unpartitionedDataset.numberOfPredictorsPlus1+1; 
						numberOfExamples < unpartitionedDataset.numberOfAllTrainingExamples-1; 
						numberOfExamples++) 
				{
					LinearRegressor lr = new LinearRegressor( new LinearRegressorDataset(unpartitionedDataset, numberOfExamples), runNumber);

					futureQueue.add(executorService.submit(new DerivativeSolverTask(lr, numberOfExamples)));

					for (UpdateRule updateRule : updateRules) {
						for (double learningRate : learningRates) {
							for (double lambda : lambdas) {
								futureQueue.add(executorService.submit(new GradientDescentTask(new GradientDescentParameters(runNumber, lr, maxNumberOfIterations, updateRule, learningRate, lambda))));
								
								if (futureQueue.size() > 100) {
									while (futureQueue.size() > 20) {
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


}
