import dataset.Dataset;
import parameterTuning.ParameterTuningParameters;

public class Main {
	public static void main(String[] args) {
		//ParameterTuningParameters.nasaParameters.fileDirectory = ParameterTuningParameters.nasaParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");
		ParameterTuningParameters.crimeCommunitiesParameters.fileDirectory = ParameterTuningParameters.crimeCommunitiesParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");
		ParameterTuningParameters.powerPlantParameters.fileDirectory = ParameterTuningParameters.powerPlantParameters.fileDirectory.replace("comp520-linearRegression", "GBmWithVariableShrinkage");

		//LinearRegressor lr = new LinearRegressor(new Dataset(ParameterTuningParameters.nasaParameters, ParameterTuningParameters.TRAINING_SAMPLE_FRACTION));
		LinearRegressor lr2 = new LinearRegressor(new Dataset(ParameterTuningParameters.powerPlantParameters, ParameterTuningParameters.TRAINING_SAMPLE_FRACTION));
		LinearRegressor lr3= new LinearRegressor(new Dataset(ParameterTuningParameters.crimeCommunitiesParameters, ParameterTuningParameters.TRAINING_SAMPLE_FRACTION));

	}
}
