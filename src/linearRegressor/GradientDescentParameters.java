package linearRegressor;
import dataset.LinearRegressorDataset;

public class GradientDescentParameters {
		//public int runNumber;
		public int maxNumberOfIterations;
		public UpdateRule updateRule;
		public double learningRate;
		public double lambda;
		public String datasetMinimalName;
		public String fileNamePrefix;
		public String subDirectory;
		public String prettyPrintOut;
		public LinearRegressorDataset dataset;
		//public String foldId;
		public String rootSubDirectory;
		public GradientDescentParameters(String rootSubDirectory, LinearRegressorDataset dataset, int maxNumberOfIterations, UpdateRule updateRule, double learningRate, double lambda) {
			this.rootSubDirectory = rootSubDirectory;
			this.maxNumberOfIterations = maxNumberOfIterations;
			this.updateRule = updateRule;
			this.learningRate = learningRate;
			this.lambda = lambda;
			this.datasetMinimalName = dataset.dataset.parameters.minimalName;
			this.dataset = dataset;
			
			this.fileNamePrefix = getFileNamePrefix();
			this.subDirectory = getSubDirectory();
			this.prettyPrintOut = getLineSeparatedPrintout();
		}
		
		private String getFileNamePrefix() {
			return String.format("%s-%fLR-%fLambda-%dTrainingExamples", 
					updateRule.name(), 
					learningRate, 
					lambda,
					dataset.numberOfTrainingExamples);
		}
		public String getMinimalDescription() {
			return String.format("[%s][LR%f][Lamda%f][NoTE%d]", 
					updateRule.name(), 
					learningRate, 
					lambda,
					dataset.numberOfTrainingExamples);
		}
		
		public String getGradientDescentLatexCaptionPrefix() {
			return String.format("%s - %s Update, %f LR, %f Lamda", 
					dataset.dataset.parameters.fullName, 
					updateRule.name(), 
					learningRate, 
					lambda);
		}
		public String getGradientDescentLatexFigureIdPrefix() {
			return String.format("%s%sUpdate%fLR%fLamda", 
					dataset.dataset.parameters.fullName, 
					updateRule.name(), 
					learningRate, 
					lambda);
		}
		
		private String getSubDirectory() {
			return rootSubDirectory + String.format("%s/%fLR/%fLambda/%dTrainingExamples/", 
					updateRule.name(),
					learningRate,
					lambda,
					dataset.numberOfTrainingExamples 
				);
		}
		
		private String getLineSeparatedPrintout() {
			return String.format("DatasetName: %s\n" +
					"UpdateRule: %s\n" + 
					"LearningRate: %f\n" +
					"Lambda: %f\n" +
					"NumberOfTrainingSamples: %d\n" +
					"NumberOfValidationSamples: %d\n" +
					"NumberOfTestSamples: %d\n" +
					"NumberOfWeights: %d\n" +
					"MaxIterations: %d\n",
					dataset.dataset.parameters.fullName,
					updateRule.name(),
					learningRate,
					lambda,
					dataset.numberOfTrainingExamples,
					dataset.numberOfValidationExamples,
					dataset.numberOfTestExamples,
					dataset.numberOfPredictorsPlus1,
					maxNumberOfIterations);
		}
		
		public static String getTabSeparatedMinimalPrintoutHeader() {
			return String.format("%s\t" + 
					"%s\t" +
					"%s\t",
					"UpdateRule",
					"LearningRate",
					"Lambda");
		}
		public String getTabSeparatedMinimalPrintout() {
			return String.format(
					"%s\t" + 
					"%f\t" +
					"%f\t",
					updateRule.name(),
					learningRate,
					lambda);
		}
}
