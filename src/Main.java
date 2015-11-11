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
		
		DatasetParameters[] datasets = new DatasetParameters[] {ParameterTuningParameters.nasaParameters, ParameterTuningParameters.powerPlantParameters, ParameterTuningParameters.crimeCommunitiesParameters};
		for (DatasetParameters dsParam : datasets) {
			for (int i = 0; i < NUMBER_OF_RUNS; i++) {
				LinearRegressorDataset lrDataset = new LinearRegressorDataset(new Dataset(dsParam, ParameterTuningParameters.TRAINING_SAMPLE_FRACTION));
				/**
				 * Loop from a training set with numberOfPredictorsPlus1 examples, validation set with numberOfAllTrainingExamples examples
				 * 						set with numberOfAllTrainingExamples-1 examples, validation set with 1 example.
				 * 
				 * Training set must minimally have as many rows as it does columns in order for the inverse calculation to work. Throws rank deficient error otherwise.
				 */
				StopWatch globalTimer = new StopWatch(), timer = new StopWatch();
				globalTimer.start();
				for (int numberOfExamples = lrDataset.numberOfPredictorsPlus1; numberOfExamples < lrDataset.numberOfAllTrainingExamples-1; numberOfExamples++) {
					timer.start();
					LinearRegressor lr = new LinearRegressor(lrDataset.getFirstNExamplesInDataset(numberOfExamples));
					lr.setLearningRateAndLambda(0.001, 0.1);
					String message = lr.runGradientDescentAndSaveResults(100000, UpdateRule.AdaptedLR);
					timer.printMessageWithTime(String.format("[%s] " + message, dsParam.minimalName));
				}
				globalTimer.printMessageWithTime("Looped through all parameters");
				
			}

		}
	}
}
