import parameterTuning.ParameterTuningParameters;
import utilities.StopWatch;
import dataset.Dataset;
import dataset.DatasetParameters;

public class Main {
	private static int NUMBER_OF_RUNS = 1;
	
	public static void main(String[] args) {
		ParameterTuningParameters.nasaParameters.fileDirectory = ParameterTuningParameters.nasaParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");
		ParameterTuningParameters.crimeCommunitiesParameters.fileDirectory = ParameterTuningParameters.crimeCommunitiesParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");
		ParameterTuningParameters.powerPlantParameters.fileDirectory = ParameterTuningParameters.powerPlantParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");
		
		DatasetParameters[] datasets = new DatasetParameters[] {/*ParameterTuningParameters.nasaParameters, */ParameterTuningParameters.powerPlantParameters, ParameterTuningParameters.crimeCommunitiesParameters};
		UpdateRule[] updateRules = new UpdateRule[] {UpdateRule.Original, UpdateRule.AdaptedLR, UpdateRule.OriginalWithRegularization, UpdateRule.AdaptedLRWithRegularization};
		double[] learningRates = new double[] {0.0001, 0.001, 0.01, 0.1, .5, 1};
		double[] lambdas = new double[] {0, 0.0001, 0.001, 0.01, 0.1, .5, 1, 5};
		for (DatasetParameters dsParam : datasets) {
			for (int i = 0; i < NUMBER_OF_RUNS; i++) {
				LinearRegressorDataset unpartitionedDataset = new LinearRegressorDataset(new Dataset(dsParam, ParameterTuningParameters.TRAINING_SAMPLE_FRACTION));
				/**
				 * Loop from a training set with numberOfPredictorsPlus1 examples, validation set with numberOfAllTrainingExamples examples
				 * 						set with numberOfAllTrainingExamples-1 examples, validation set with 1 example.
				 * 
				 * Training set must minimally have as many rows as it does columns in order for the inverse calculation to work. Throws rank deficient error otherwise.
				 */
				StopWatch globalTimer = new StopWatch(), timer = new StopWatch();
				globalTimer.start();
				for (int numberOfExamples = unpartitionedDataset.numberOfPredictorsPlus1+1; 
						numberOfExamples < unpartitionedDataset.numberOfAllTrainingExamples-1; 
						numberOfExamples++) 
				{
					LinearRegressorDataset lrDataset = new LinearRegressorDataset(unpartitionedDataset, numberOfExamples);
					LinearRegressor lr = new LinearRegressor(lrDataset);
					timer.start();
					String message = lr.calculateAndSaveOptimalWeightsBySolvingDerivative();
					timer.printMessageWithTime(String.format("[%s] " + message, dsParam.minimalName));
					for (UpdateRule updateRule : updateRules) {
						for (double learningRate : learningRates) {
							if (updateRule == UpdateRule.AdaptedLRWithRegularization || updateRule == UpdateRule.OriginalWithRegularization) {
								for (double lambda : lambdas) {
									timer.start();
									message = lr.runGradientDescentAndSaveResults(100000, updateRule, learningRate, lambda);
									timer.printMessageWithTime(String.format("[%s] " + message, dsParam.minimalName));
								}
							} else {
								timer.start();
								message = lr.runGradientDescentAndSaveResults(100000, updateRule, learningRate, 0);
								timer.printMessageWithTime(String.format("[%s] " + message, dsParam.minimalName));
							}
						}
					}
				}
				globalTimer.printMessageWithTime("Looped through all parameters");
			}
		}
	}
}
