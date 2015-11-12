
public class GradientDescentParameters {
		public int runNumber;
		public int maxNumberOfIterations;
		public UpdateRule updateRule;
		public double learningRate;
		public double lambda;
		public LinearRegressor lr;
		public String datasetMinimalName;
		public String fileNamePrefix;
		public String subDirectory;
		public String prettyPrintOut;
		
		public GradientDescentParameters(int runNumber, LinearRegressor lr, int maxNumberOfIterations, UpdateRule updateRule, double learningRate, double lambda) {
			this.runNumber = runNumber;
			this.lr = lr;
			this.maxNumberOfIterations = maxNumberOfIterations;
			this.updateRule = updateRule;
			this.learningRate = learningRate;
			this.lambda = lambda;
			this.datasetMinimalName = lr.dataset.dataset.parameters.minimalName;
			this.fileNamePrefix = getFileNamePrefix();
			this.subDirectory = getSubDirectory();
			this.prettyPrintOut = getLineSeparatedPrintout();
			
		}
		
		private String getFileNamePrefix() {
			return String.format("%s-run%d-gradientDescent-%dMaxIterations-%s-%fLR-%fLambda-%dTrainingExamples", 
					lr.dataset.dataset.parameters.minimalName, 
					runNumber,
					maxNumberOfIterations, 
					updateRule.name(), 
					learningRate, 
					lambda,
					lr.dataset.numberOfTrainingExamples);
		}
		
		private String getSubDirectory() {
			return String.format("%s/Run%d/gradientDescent/%dMaxIterations/%s/%fLR/%fLambda/%dTrainingExamples/", 
					lr.dataset.dataset.parameters.minimalName,
					runNumber,
					maxNumberOfIterations, 
					updateRule.name(),
					learningRate,
					lambda,
					lr.dataset.numberOfTrainingExamples 
				);
		}
		
		private String getLineSeparatedPrintout() {
			return String.format("DatasetName: %s\n" +
					"RunNumber: %d\n" +
					"MaxIterations: %d\n" +
					"UpdateRule: %s\n" + 
					"LearningRate: %f\n" +
					"Lambda: %f\n" +
					"NumberOfTrainingSamples: %d\n" +
					"NumberOfValidationSamples: %d\n",
					
					lr.dataset.dataset.parameters.fullName,
					runNumber,
					maxNumberOfIterations,
					updateRule.name(),
					learningRate,
					lambda,
					lr.dataset.numberOfTrainingExamples,
					lr.dataset.numberOfValidationExamples);
		}
}
