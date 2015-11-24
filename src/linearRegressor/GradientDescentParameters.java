package linearRegressor;
import dataset.LinearRegressorDataset;

public class GradientDescentParameters {
		public static int maxIterationsSinceTrainingErrorImproved = 20000;
		public int runNumber;
		public int maxNumberOfIterations;
		public UpdateRule updateRule;
		public double learningRate;
		public double lambda;
		public String datasetMinimalName;
		public String fileNamePrefix;
		public String subDirectory;
		public String prettyPrintOut;
		public LinearRegressorDataset dataset;
		public String foldId;
		public GradientDescentParameters(int runNumber, String foldId, LinearRegressorDataset dataset, int maxNumberOfIterations, UpdateRule updateRule, double learningRate, double lambda) {
			this.runNumber = runNumber;
			this.maxNumberOfIterations = maxNumberOfIterations;
			this.updateRule = updateRule;
			this.learningRate = learningRate;
			this.lambda = lambda;
			this.datasetMinimalName = dataset.dataset.parameters.minimalName;
			this.dataset = dataset;
			this.foldId = foldId;
			
			this.fileNamePrefix = getFileNamePrefix();
			this.subDirectory = getSubDirectory();
			this.prettyPrintOut = getLineSeparatedPrintout();

			
		}
		
		private String getFileNamePrefix() {
			return String.format("%s-run%d-%s-gradientDescent-%dMaxIterations-%s-%fLR-%fLambda-%dTrainingExamples", 
					dataset.dataset.parameters.minimalName, 
					runNumber,
					foldId,
					maxNumberOfIterations, 
					updateRule.name(), 
					learningRate, 
					lambda,
					dataset.numberOfTrainingExamples);
		}
		public String getMinimalDescription() {
			return String.format("[%s][Run%d][%s][%s][LR%f][Lamda%f][NoTE%d]", 
					dataset.dataset.parameters.minimalName, 
					runNumber,
					foldId,
					updateRule.name(), 
					learningRate, 
					lambda,
					dataset.numberOfTrainingExamples);
		}
		
		private String getSubDirectory() {
			return String.format("%s/Run%d/%s/gradientDescent/%dMaxIterations/%s/%fLR/%fLambda/%dTrainingExamples/", 
					dataset.dataset.parameters.minimalName,
					runNumber,
					foldId,
					maxNumberOfIterations, 
					updateRule.name(),
					learningRate,
					lambda,
					dataset.numberOfTrainingExamples 
				);
		}
		
		private String getLineSeparatedPrintout() {
			return String.format("DatasetName: %s\n" +
					"RunNumber: %d\n" +
					"FoldId: %s\n" +
					"UpdateRule: %s\n" + 
					"LearningRate: %f\n" +
					"Lambda: %f\n" +
					"NumberOfTrainingSamples: %d\n" +
					"NumberOfValidationSamples: %d\n" +
					"NumberOfTestSamples: %d\n" +
					"NumberOfWeights: %d\n" +
					"MaxIterations: %d\n" +
					"MaxIterationsSinceTrainingErrorImproved: %d\n",
					dataset.dataset.parameters.fullName,
					runNumber,
					foldId,
					updateRule.name(),
					learningRate,
					lambda,
					dataset.numberOfTrainingExamples,
					dataset.numberOfValidationExamples,
					dataset.numberOfTestExamples,
					dataset.numberOfPredictorsPlus1,
					maxNumberOfIterations,
					maxIterationsSinceTrainingErrorImproved);
		}
}
