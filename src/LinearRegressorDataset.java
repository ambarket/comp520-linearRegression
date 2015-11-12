import java.util.Arrays;

import Jama.Matrix;


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
	
	public Dataset dataset;
	
	public LinearRegressorDataset(Dataset dataset) {
		// Automatically adds extra column to the X matrices for the first weight
		allTrainX = ExtraMatrixMethods.convertAttributeArrayToMatrix(dataset.getTrainingInstances());
		allTrainY = ExtraMatrixMethods.convertAttributeArrayToMatrix(dataset.getTrainingResponses());
		testX = ExtraMatrixMethods.convertAttributeArrayToMatrix(dataset.getTestInstances());
		testY = ExtraMatrixMethods.convertAttributeArrayToMatrix(dataset.getTestResponses());
		
		numberOfAllTrainingExamples = allTrainX.getRowDimension();
		numberOfPredictorsPlus1 = allTrainX.getColumnDimension();
		numberOfTestExamples = testX.getRowDimension();
		this.dataset = dataset;
	}
	
	/**
	 * Select numberOfTrainingExamples rows uniformly at random from original to use as trainX/trainY. 
	 * The remaining examples become validX/validY.
	 * @param original
	 * @param numberOfTrainingExamples
	 */
	public LinearRegressorDataset (LinearRegressorDataset original, int numberOfTrainingExamples) {
		this.numberOfTrainingExamples = numberOfTrainingExamples;
		this.numberOfValidationExamples = original.numberOfAllTrainingExamples - numberOfTrainingExamples;
		this.numberOfTestExamples = original.numberOfTestExamples;
		this.numberOfPredictorsPlus1 = original.numberOfPredictorsPlus1;
		this.numberOfAllTrainingExamples = original.numberOfAllTrainingExamples;
		this.dataset = original.dataset;
		this.allTrainX = original.allTrainX;
		this.allTrainY = original.allTrainY;
		this.testX = original.testX;
		this.testY = original.testY;
		
		int[] shuffledRows = new RandomSample().fisherYatesShuffle(numberOfAllTrainingExamples);
		int[] trainingRows = Arrays.copyOfRange(shuffledRows, 0, numberOfTrainingExamples-1);
		int[] validationRows = Arrays.copyOfRange(shuffledRows, numberOfTrainingExamples, numberOfAllTrainingExamples-1);
		
		this.trainX = allTrainX.getMatrix(trainingRows, 0, original.numberOfPredictorsPlus1-1);
		this.trainY = allTrainY.getMatrix(trainingRows, 0, 0);
		this.validX = allTrainX.getMatrix(validationRows, 0, original.numberOfPredictorsPlus1-1);
		this.validY = allTrainY.getMatrix(validationRows, 0, 0);

		

	}
}
