package linearRegressor;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.Callable;

import Jama.Matrix;
import dataset.DatasetParameters;
import utilities.ExtraMatrixMethods;
import utilities.SimpleHostLock;
import utilities.StopWatch;


public class DerivativeSolverTask implements Callable<Void> {
	public int numberOfExamples;
	LinearRegressor lr;
	public int runNumber;
	public DerivativeSolverTask(LinearRegressor lr, int numberOfExamples, int runNumber) {
		this.lr = lr;
		this.numberOfExamples = numberOfExamples;
		this.runNumber = runNumber;
	}
	public Void call() {
		StopWatch timer = new StopWatch().start();
		
		String directory = String.format("%s/%s/Run%d/derivativeSetToZero/%dTrainingExamples/", Main.RESULTS_DIRECTORY, lr.dataset.dataset.parameters.minimalName,runNumber,  numberOfExamples);
		new File(directory).mkdirs();
		if (SimpleHostLock.checkDoneLock(directory + "doneLock.txt")) {
			timer.printMessageWithTime(String.format("[%s] [Run %d] Already done by another host %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, runNumber, numberOfExamples));
			return null;
		}
		if (!SimpleHostLock.checkAndClaimHostLock(directory + "hostLock.txt")) {
			timer.printMessageWithTime(String.format("[%s] [Run %d] Claimed by another host %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, runNumber, numberOfExamples));
			return null;
		}
		
		timer.printMessageWithTime(String.format("[%s] [Run %d] Starting %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, runNumber, numberOfExamples));
		
		Matrix w = lr.dataset.trainX.inverse().times(lr.dataset.trainY);

		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(directory + numberOfExamples + "TrainingExamples-optimalWeightsBySolvingDerivative.txt"));
			bw.write(String.format("Weights: %s\n", ExtraMatrixMethods.convertWeightsToTabSeparatedString(w)));
			bw.write(String.format("TrainingRMSE: %f\n", lr.getRMSE(lr.dataset.trainX, lr.dataset.trainY, w)));
			bw.write(String.format("ValidationRMSE: %f\n", lr.getRMSE(lr.dataset.validX, lr.dataset.validY, w)));
			bw.write(String.format("TestRMSE: %f\n", lr.getRMSE(lr.dataset.testX, lr.dataset.testY, w)));
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
			timer.printMessageWithTime(String.format("[%s] [Run %d] Failed to write done lock %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, runNumber, numberOfExamples));
		}
		SimpleHostLock.writeDoneLock(directory + "doneLock.txt");
		timer.printMessageWithTime(String.format("[%s] [Run %d] Successfully finished %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, runNumber, numberOfExamples));
		return null;
	}
	
	public static LearningCurveEntry readOptimalWeightsBySolvingDerivate(DatasetParameters datasetParams, int numberOfExamples, int runNumber) {
		String directory = String.format("%s/%s/Run%d/derivativeSetToZero/%dTrainingExamples/", Main.RESULTS_DIRECTORY, datasetParams.minimalName,runNumber,  numberOfExamples);
		BufferedReader br;
		LearningCurveEntry result = new LearningCurveEntry();
		result.numberOfTrainingExamples = numberOfExamples;
		try {
			br = new BufferedReader(new FileReader(directory + numberOfExamples + "TrainingExamples-optimalWeightsBySolvingDerivative.txt"));
			br.readLine();
			result.trainingError = Double.parseDouble(br.readLine().split(": ")[1]);
			result.validationError = Double.parseDouble(br.readLine().split(": ")[1]);
			result.testError = Double.parseDouble(br.readLine().split(": ")[1]);
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return result;
	}
}

