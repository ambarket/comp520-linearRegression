package utilities;


import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;

public class MathematicaListCreator {
	
	public static String verticalLineAtXValueWithHeight(double x, double height) {
		return "{{" + x + ", " + 0 +"}, {" + x + ", " + height + "}}\n";
	}
	public static String convertToMathematicaList(ArrayList<Double> error, int indexOffset) {
		StringBuffer retval = new StringBuffer();
		retval.append("{\n");
		for (int i = 0; i < error.size(); i++) {
			retval.append(String.format("\t{ %d, %.5f }", i+indexOffset, error.get(i)));
			if (i != error.size() - 1) {
				retval.append(",");
			} 
			retval.append("\n");
		}
		retval.append("}");
		return retval.toString();
	}
	
	public static void convertToMathematicaList(ArrayList<Double> error, int indexOffset, int step, BufferedWriter bw) throws IOException {
		bw.append("{\n");
		for (int i = 0; i < error.size(); i+=step) {
			bw.append(String.format("\t{ %d, %.5f }", i+indexOffset, error.get(i)));
			if (i+step <= error.size() - 1) {
				bw.append(",");
			} 
			bw.append("\n");
		}
		bw.append("}");
	}
	
	public static String convertToMathematicaList(ArrayList<Double> error) {
		return convertToMathematicaList(error, 1);
	}
	
	public static void convertToMathematicaList(ArrayList<Double> error, int step, BufferedWriter bw) throws IOException {
		convertToMathematicaList(error, 1, step, bw);
	}
	
	public static String convertToMathematicaList(double[] error) {
		StringBuffer retval = new StringBuffer();
		retval.append("{\n");
		for (int i = 0; i < error.length; i++) {
			retval.append(String.format("\t{ %d, %.5f }", i+1, error[i]));
			if (i != error.length - 1) {
				retval.append(",");
			} 
			retval.append("\n");
		}
		retval.append("}");
		return retval.toString();
	}
	
	public static String convertNObjectsIntoNDimensionalListEntry(Object... args) {

		StringBuffer retval = new StringBuffer();
		retval.append("{");
		for (int i = 0; i < args.length-1; i++) {
			double doub = Double.MIN_NORMAL;
			try {
				doub = (double)args[i];
			} catch (ClassCastException e) {
				
			}
			if (DoubleCompare.equals(doub, Double.MIN_VALUE)) {
				retval.append(args[i].toString() + ", ");
			} else {
				retval.append(String.format("%f", doub) + ", ");
			}
		}
		double doub = Double.MIN_NORMAL;
		try {
			doub = (double)args[args.length-1];
		} catch (ClassCastException e) {
			
		}
		if (DoubleCompare.equals(doub, Double.MIN_VALUE)) {
			retval.append(args[args.length-1].toString() + "}");
		} else {
			retval.append(String.format("%f", doub) + "}");
		}
		
		return retval.toString();
	}
}
