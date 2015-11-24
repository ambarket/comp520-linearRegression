package utilities;
import Jama.Matrix;
import dataset.Attribute;


public class ExtraMatrixMethods {
	
	public static Matrix averageColumnVector(Matrix... matrices) {
		int columns = matrices[0].getColumnDimension();
		if (matrices[0].getRowDimension() != 1) throw new IllegalArgumentException();
		//double[] avg = new double[columns];
		Matrix avg = new Matrix(1, columns);
		
		for (Matrix m : matrices) {
			avg.plusEquals(m);
		}
		avg.timesEquals(1.0 / matrices.length);
		return avg;
	}
	
    public static double[] add(double[] A, double[] B) {
        assert(A.length == B.length);
    	int m = A.length;
        double[] C = new double[m];
        for (int i = 0; i < m; i++)
        	C[i] = A[i] + B[i];
        return C;
    }
    
    // Adds cooresponding elements of B to A. A is modified and returned as a result
    public static double[] addToInPlace(double[] A, double[] B) {
        assert(A.length == B.length);
    	int m = A.length;
        for (int i = 0; i < m; i++)
        	A[i] += B[i];
        return A;
    }
    
	
	public static double[] multiplyScalar(double scalar, double[] vector) {
		double[] retval = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			retval[i] = scalar * vector[i];
		}
		return retval;
	}
	public static double[] multiplyScalar(double scalar, Attribute[] vector) {
		double[] retval = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			retval[i] = scalar * vector[i].getNumericValue();
		}
		return retval;
	}
	
	public static String convertWeightsToTabSeparatedString(Matrix w) {
		StringBuffer retval = new StringBuffer();
		for (int i = 0; i < w.getRowDimension(); i++) {
			if (i != 0) {
				retval.append("\t");
			}
			retval.append(String.format("%f", w.get(i,0)));
		}
		return retval.toString();
	}
	
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
