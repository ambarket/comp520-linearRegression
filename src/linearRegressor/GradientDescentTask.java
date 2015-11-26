package linearRegressor;
import java.io.File;
import java.util.concurrent.Callable;

import utilities.SimpleHostLock;
import utilities.StopWatch;


public class GradientDescentTask implements Callable<Void> {
	GradientDescentParameters parameters;
	LinearRegressor lr;
	int foldNumber;
	String locksDirectory;
	public GradientDescentTask(LinearRegressor lr, GradientDescentParameters parameters) {
		this.parameters = parameters;
		this.lr = lr;
		this.locksDirectory = Main.LOCKS_DIRECTORY + parameters.subDirectory;
		new File(locksDirectory).mkdirs();
	}
	
	@Override
	public Void call() throws Exception {
		StopWatch timer = new StopWatch().start();
		timer.printMessageWithTime(String.format("[%s] Starting gradientDescent on %s" , parameters.datasetMinimalName, parameters.subDirectory));
		GradientDescentInformation info = null;
		String message = checkDoneAndHostLocks();
		if (message == null) {
			info = lr.runGradientDescent(parameters);
			info.saveToFile();
			info.generateErrorCurveScript();
			info.executeErrorCurveScript();
			message = writeDoneLock();
		}
		timer.printMessageWithTime(String.format("[%s] " + message + " %s ", parameters.datasetMinimalName, parameters.subDirectory));
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
