package linearRegressor;
import java.io.File;
import java.util.concurrent.Callable;

import dataset.LinearRegressorDataset;
import utilities.SimpleHostLock;
import utilities.StopWatch;


public class AverageGradientDescentRunDataTask implements Callable<Void> {
	GradientDescentParameters globalParameters;
	LinearRegressorDataset lrDataset;
	int NUMBER_OF_RUNS;
	int submissionNumber, totalSubmissions;
	String locksDirectory;
	public AverageGradientDescentRunDataTask(LinearRegressorDataset lrDataset, int NUMBER_OF_RUNS, int submissionNumber, int totalSubmissions, GradientDescentParameters parameters) {
		this.globalParameters = parameters;
		this.locksDirectory = Main.LOCKS_DIRECTORY + parameters.subDirectory;
		new File(locksDirectory).mkdirs();

		this.submissionNumber = submissionNumber;
		this.totalSubmissions = totalSubmissions;
		this.NUMBER_OF_RUNS = NUMBER_OF_RUNS;
		this.lrDataset = lrDataset;
	}
	
	@Override
	public Void call() throws Exception {
		StopWatch timer = new StopWatch().start();
		timer.printMessageWithTime(String.format("[%s] [AVERAGES] [Test %d/%d] Starting averaging gradientDescent on %s" , lrDataset.dataset.parameters.minimalName, submissionNumber, totalSubmissions, globalParameters.subDirectory));
		String message = checkDoneAndHostLocks();
		if (message == null) {
			GradientDescentInformation[] infos = new GradientDescentInformation[NUMBER_OF_RUNS];
			for (int runNumber = 0; runNumber < NUMBER_OF_RUNS; runNumber++) {
				
				String resultsSubDirectory = String.format("%s/gradientDescent/Run%d/%s/", 
						lrDataset.dataset.parameters.minimalName,
						runNumber,
						"StandardValidation"
					);
				GradientDescentParameters parameters = new GradientDescentParameters(resultsSubDirectory,
						lrDataset, 
						globalParameters.maxNumberOfIterations, 
						globalParameters.updateRule, 
						globalParameters.learningRate, 
						globalParameters.lambda);

				infos[runNumber] = GradientDescentInformation.readFromFile(parameters);
				System.out.println("Done with run " + runNumber);
			}
			GradientDescentInformation avg = GradientDescentInformation.averageInformation(globalParameters, infos);
			avg.saveToFile();
			avg.generateErrorCurveScript();
			avg.executeErrorCurveScript();
		}
		message = writeDoneLock();

		timer.printMessageWithTime(String.format("[%s] [AVERAGES] [Test %d/%d] Finished averaging gradientDescent on %s" , lrDataset.dataset.parameters.minimalName, submissionNumber, totalSubmissions, globalParameters.subDirectory));
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
