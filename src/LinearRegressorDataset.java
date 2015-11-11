import Jama.Matrix;
import dataset.Dataset;


public class LinearRegressorDataset {
	int numberOfAllTrainingExamples;
	int numberOfTrainingExamples;
	int numberOfValidationExamples;
	int numberOfTestExamples;
	int numberOfPredictorsPlus1;
	
	public Matrix allTrainX;
	public Matrix allTrainY;
	
	public Matrix trainX;
	public Matrix trainY;
	public Matrix validX;
	public Matrix validY;
	public Matrix testX;
	public Matrix testY;
	
	public LinearRegressorDataset(Dataset dataset) {
		// Automatically adds extra column to the X matrices for the first weight
		allTrainX = ExtraMatrixMethods.convertAttributeArrayToMatrix(dataset.getTrainingInstances());
		allTrainY = ExtraMatrixMethods.convertAttributeArrayToMatrix(dataset.getTrainingResponses());
		testX = ExtraMatrixMethods.convertAttributeArrayToMatrix(dataset.getTestInstances());
		testY = ExtraMatrixMethods.convertAttributeArrayToMatrix(dataset.getTestResponses());
		
		numberOfAllTrainingExamples = allTrainX.getRowDimension();
		numberOfPredictorsPlus1 = allTrainX.getColumnDimension();
		numberOfTestExamples = testX.getRowDimension();
	}
	
	private LinearRegressorDataset(LinearRegressorDataset original, Matrix trainX, Matrix trainY, Matrix validX, Matrix validY) {
		this.trainX = trainX;
		this.trainY = trainY;
		this.validX = validX;
		this.validY = validY;
		this.testX = original.testX;
		this.testY = original.testY;
		numberOfTrainingExamples = trainX.getRowDimension();
		numberOfValidationExamples = validX.getRowDimension();
		numberOfTestExamples = original.numberOfTestExamples;
		numberOfPredictorsPlus1 = original.numberOfPredictorsPlus1;
		numberOfAllTrainingExamples = original.numberOfAllTrainingExamples;
	}
	
	public LinearRegressorDataset getFirstNExamplesInDataset(int numberOfExamples) {
		return new LinearRegressorDataset(this,
				allTrainX.getMatrix(0, numberOfExamples-1, 0, numberOfPredictorsPlus1-1),
				allTrainY.getMatrix(0, numberOfExamples-1, 0, 0),
				allTrainX.getMatrix(numberOfExamples, numberOfAllTrainingExamples-1, 0, numberOfPredictorsPlus1-1),
				allTrainY.getMatrix(numberOfExamples, numberOfAllTrainingExamples-1, 0, 0));
				
	}
}
