package linearRegressor;
import java.io.File;
import java.util.concurrent.Callable;

import utilities.SimpleHostLock;
import utilities.StopWatch;


public class GradientDescentTask implements Callable<Void> {
	GradientDescentParameters parameters;
	LinearRegressor lr;
	int runNumber;
	int submissionNumber, totalSubmissions;
	String locksDirectory;
	public GradientDescentTask(LinearRegressor lr, int runNumber, int submissionNumber, int totalSubmissions, GradientDescentParameters parameters) {
		this.parameters = parameters;
		this.lr = lr;
		this.locksDirectory = Main.LOCKS_DIRECTORY + parameters.subDirectory;
		new File(locksDirectory).mkdirs();
		this.runNumber = runNumber;
		this.submissionNumber = submissionNumber;
		this.totalSubmissions = totalSubmissions;
	}
	
	@Override
	public Void call() throws Exception {
		StopWatch timer = new StopWatch().start();
		timer.printMessageWithTime(String.format("[%s] [Run%d] [Test %d/%d] Starting gradientDescent on %s" , parameters.datasetMinimalName, runNumber, submissionNumber, totalSubmissions, parameters.subDirectory));
		GradientDescentInformation info = null;
		String message = checkDoneAndHostLocks();
		if (message == null) {
			info = lr.runGradientDescent(parameters, runNumber, submissionNumber, totalSubmissions);
			info.saveToFile();
			//info.generateErrorCurveScript();
			//info.executeErrorCurveScript();
			message = writeDoneLock();
		}
		timer.printMessageWithTime(String.format("[%s] [Run%d] [Test %d/%d]" + message + " %s ", parameters.datasetMinimalName, runNumber, submissionNumber, totalSubmissions, parameters.subDirectory));
		return null;
	}
	
	
	
	public String checkDoneAndHostLocks() {
		if (SimpleHostLock.checkDoneLock(locksDirectory + "doneLock.txt")) {
			return "Completed by another host";
		}
		if (!SimpleHostLock.checkAndClaimHostLock(locksDirectory + "hostLock.txt")) {
			return "Claimed by another host";
		}
		return null;
	}
	
	public String writeDoneLock() {
		if (SimpleHostLock.writeDoneLock(locksDirectory + "doneLock.txt")) {
			return "Gradient descent completed";
		} else {
			return "Failed to write done lock for some reason.";
		}
	}
}
