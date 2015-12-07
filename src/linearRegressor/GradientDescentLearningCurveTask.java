package linearRegressor;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.Callable;

import utilities.SimpleHostLock;
import utilities.StopWatch;


public class GradientDescentLearningCurveTask implements Callable<Void> {
	GradientDescentParameters parameters;
	LinearRegressor lr;
	int runNumber, submissionNumber, totalSubmissions;
	String locksDirectory;
	
	private static String learningCurveEntryFileName = "learningCurveEntry.txt";
	
	public GradientDescentLearningCurveTask(LinearRegressor lr, int runNumber, int submissionNumber, int totalSubmissions, GradientDescentParameters parameters) {
		this.parameters = parameters;
		this.lr = lr;
		this.locksDirectory = Main.LOCKS_DIRECTORY + parameters.subDirectory;
		new File(locksDirectory).mkdirs();
	}
	
	@Override
	public Void call() throws Exception {
		StopWatch timer = new StopWatch().start();
		timer.printMessageWithTime(String.format("[%s] [Run%d] [Test %d/%d] Starting gradientDescentLearningCurveTask on %s" , parameters.datasetMinimalName, runNumber, submissionNumber, totalSubmissions, parameters.subDirectory));

		GradientDescentInformation info = null;
		String message = checkDoneAndHostLocks();
		if (message == null) {
			info = lr.runGradientDescent(parameters, runNumber, submissionNumber, totalSubmissions);
			info.summary.saveToFile();
			saveToLearningCurveFile(info.summary);
			message = writeDoneLock();
		}
		timer.printMessageWithTime(String.format("[%s] [Run%d] [Test %d/%d]" + message + " %s ", parameters.datasetMinimalName, runNumber, submissionNumber, totalSubmissions, parameters.subDirectory));
		return null;
	}
	
	public void saveToLearningCurveFile(GradientDescentSummary summary) {	
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(summary.directory + learningCurveEntryFileName));
			bw.write(String.format("NumberOfTrainingExamples\t"
					+ "MinTrainingError\t"
					+ "MinValidationError\t"
					+ "MinTestError\n"));
			bw.write(String.format("%d\t%f\t%f\t%f\n", 
					summary.parameters.dataset.numberOfTrainingExamples, 
					summary.validationStoppingTrainingError, 
					summary.validationStoppingValidationError, 
					summary.validationStoppingTestError));
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	public static LearningCurveEntry readLearningCurveFile(GradientDescentParameters parameters) {	
		LearningCurveEntry retval = new LearningCurveEntry();
		try {
			BufferedReader br = new BufferedReader(new FileReader(Main.RESULTS_DIRECTORY + parameters.subDirectory + learningCurveEntryFileName));
			br.readLine();
			
			String[] components = br.readLine().split("\t");
			retval.numberOfTrainingExamples = Integer.parseInt(components[0]);
			retval.trainingError = Double.parseDouble(components[1]);
			retval.validationError = Double.parseDouble(components[2]);
			retval.testError = Double.parseDouble(components[3]);
			br.close();
		} catch (FileNotFoundException e) {
			System.out.print("File Not found " + parameters.subDirectory + learningCurveEntryFileName);
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return retval;
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
