package linearRegressor;
import java.util.concurrent.Callable;

import utilities.StopWatch;


public class GradientDescentTask implements Callable<Void> {
	GradientDescentParameters parameters;
	LinearRegressor lr;
	int foldNumber;
	public GradientDescentTask(LinearRegressor lr, GradientDescentParameters parameters) {
		this.parameters = parameters;
		this.lr = lr;
	}
	
	@Override
	public Void call() throws Exception {
		StopWatch timer = new StopWatch().start();
		timer.printMessageWithTime(String.format("[%s] Starting gradientDescent on %s" , parameters.datasetMinimalName, parameters.subDirectory));
		String message = lr.runGradientDescentAndSaveResults(parameters);
		timer.printMessageWithTime(String.format("[%s] " + message + " %s ", parameters.datasetMinimalName, parameters.subDirectory));
		return null;

	}
}
