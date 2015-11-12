import Jama.Matrix;


public class ExtraMatrixMethods {
	public static Matrix getUnitVector(Matrix m) {
		return m.times(1.0 / getL2Norm(m));
	}
	
	public static double getL2Norm(Matrix m) {
		return Math.sqrt(getSumOfSquares(m));
	}
	
	public static double getSumOfSquares(Matrix m) {
		if (m.getColumnDimension() != 1) {
			System.out.println("SumOFSquares only defined on vectors");
		}
		double sumOfSquares = 0.0;
		double[] elements = m.getColumnPackedCopy();
		for (int i = 0; i < elements.length; i++) {
			sumOfSquares += elements[i] * elements[i];
		}
		return sumOfSquares;
	}
	
	public static void printDimensions(String name, Matrix X) {
		System.out.println(String.format("%s: %d rows, %d cols", name, X.getRowDimension(), X.getColumnDimension()));
	}
	
	
	public static Matrix convertAttributeArrayToMatrix(Attribute[][] array) {
		double[][] doubleArray = new double[array.length][array[0].length+1];
		for (int i = 0; i < array.length; i++) {
			doubleArray[i][0] = 1;
			for (int j = 1; j < array[0].length+1; j++) {
				try {
					if (!array[i][j-1].isMissingValue()) {
						doubleArray[i][j] = array[i][j-1].getNumericValue();
					}
				} catch (Exception e) {
					System.out.println();
				}
			}
		}
		return new Matrix(doubleArray);
	}
	
	public static Matrix convertAttributeArrayToMatrix(Attribute[] array) {
		double[][] doubleArray = new double[array.length][1];
		for (int i = 0; i < array.length; i++) {
			doubleArray[i][0] = array[i].getNumericValue();
		}
		return new Matrix(doubleArray);
	}
}
