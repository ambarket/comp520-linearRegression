package utilities;


public class SumCountAverage {
	private double 	sum = 0.0, sumOfSquares = 0.0, mean = 0.0; 
	private int count = 0;
	private boolean meanOutOfDate = false;
	
	public double getSum() {
		return sum;
	}
	
	public double getSumOfSquares() {
		return sumOfSquares;
	}
	
	public int getCount() {
		return count;
	}
	
	public double getMean() {
		if (meanOutOfDate) {
			mean = sum / count;
		}
		return mean;
	}
	
	public double getSquaredError() {
		if (meanOutOfDate) {
			mean = sum / count;
		}
		double val = sumOfSquares - (mean * sum);
		if (val < 0) {
			val = 0;
		}
		return val;
	}
	
	public double getMeanSquaredError() {
		return getSquaredError() / count;
	}
	
	public double getRootMeanSquaredError() {
		return Math.sqrt(getMeanSquaredError());
	}
	
	public void addData(double data) {
		sum += data;
		sumOfSquares += data * data;
		count++;
		meanOutOfDate = true;
	}
	
	public void subtractData(double data) {
		sum -= data;
		sumOfSquares -= data * data;
		count--;
		meanOutOfDate = true;
	}
	
	public void addSumCountAverage(SumCountAverage data) {
		sum += data.sum;
		sumOfSquares += data.sumOfSquares;
		count += data.count;
		meanOutOfDate = true;
	}
	
	public void subtractSumCountAverage(SumCountAverage data) {
		sum -= data.sum;
		sumOfSquares -= data.sumOfSquares;
		count -= data.count;
		meanOutOfDate = true;
	}
	
	public SumCountAverage() {
		
	}
	
	public void reset() {
		sum = 0.0;
		sumOfSquares = 0.0;
		mean = 0.0; 
		count = 0;
		meanOutOfDate = false;
	}
	public SumCountAverage(double initialSum, double initialSumOfSquares, int initialCount) {
		this.sum = initialSum;
		this.sumOfSquares = initialSumOfSquares;
		this.count = initialCount;
	}
}
