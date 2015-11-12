import java.util.concurrent.Callable;


public class GradientDescentTask implements Callable<Void> {
	GradientDescentParameters parameters;
	
	public GradientDescentTask(GradientDescentParameters parameters) {
		this.parameters = parameters;
	}
	
	@Override
	public Void call() throws Exception {
		StopWatch timer = new StopWatch().start();
		timer.printMessageWithTime(String.format("[%s] Starting gradientDescent on %s" , parameters.datasetMinimalName, parameters.subDirectory));
		String message = parameters.lr.runGradientDescentAndSaveResults(parameters);
		timer.printMessageWithTime(String.format("[%s] " + message + " %s ", parameters.datasetMinimalName, parameters.subDirectory));
		return null;

	}
}
