import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.Callable;

import Jama.Matrix;


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
		
		String directory = String.format("%s/%s/Run%d/derivativeSetToZero/%dTrainingExamples/", lr.resultsDirectory, lr.dataset.dataset.parameters.minimalName,runNumber,  numberOfExamples);
		new File(directory).mkdirs();
		if (SimpleHostLock.checkDoneLock(directory + "doneLock.txt")) {
			timer.printMessageWithTime(String.format("[%s] Already done by another host %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, numberOfExamples));
			return null;
		}
		if (!SimpleHostLock.checkAndClaimHostLock(directory + "hostLock.txt")) {
			timer.printMessageWithTime(String.format("[%s] Claimed by another host %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, numberOfExamples));
			return null;
		}
		
		timer.printMessageWithTime(String.format("[%s] Starting %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, numberOfExamples));
		
		Matrix w = lr.dataset.trainX.inverse().times(lr.dataset.trainY);

		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(directory + numberOfExamples + "TrainingExamples-optimalWeightsBySolvingDerivative.txt"));
			bw.write(String.format("Weights: %s\n", LinearRegressor.convertWeightsToTabSeparatedString(w)));
			bw.write(String.format("TrainingRMSE: %f\n", lr.getRMSE(lr.dataset.trainX, lr.dataset.trainY, w)));
			bw.write(String.format("TestRMSE: %f\n", lr.getRMSE(lr.dataset.testX, lr.dataset.testY, w)));
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
			timer.printMessageWithTime(String.format("[%s] Failed to write done lock %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, numberOfExamples));
		}
		SimpleHostLock.writeDoneLock(directory + "doneLock.txt");
		timer.printMessageWithTime(String.format("[%s] Successfully finished %d training example DerivateSolver", lr.dataset.dataset.parameters.minimalName, numberOfExamples));
		return null;
	}
}
