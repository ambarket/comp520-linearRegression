package dataset;

public class Attribute implements Comparable<Attribute> {
	public enum Type {Numeric, Categorical};
	public static String MISSING_CATEGORY = "MISSING_CATEGORY";
	
	private Type type;
	private boolean missing;
	
	private Double numericValue;
	private String categoricalValue;
	
	public boolean isMissingValue() {
		return missing;
	}
	
	public Type getType() {
		return type;
	}
	
	public Attribute(Type attributeType) {
		missing = true;
		this.type = attributeType;
		if (this.type == Type.Categorical) {
			categoricalValue = MISSING_CATEGORY;
		}
	}
	
	public Attribute(Double numericValue) {
		this.numericValue = numericValue;
		if (numericValue == null) {
			missing = true;
		}
		this.type = Type.Numeric;
	}
	
	public Attribute(String stringValue) {
		this.categoricalValue = stringValue;
		if (stringValue == null) {
			missing = true;
		}
		this.type = Type.Categorical;
	}
	
	public String getCategoricalValue() {
		if (this.type != Type.Categorical) {
			throw new IllegalStateException("Attempt to get the categorical value of a numeric attribute");
		}
		return categoricalValue;
	}
	
	public Double getNumericValue() {
		if (this.type != Type.Numeric) {
			throw new IllegalStateException("Attempt to get the numeric value of a categorical attribute");
		}
		return numericValue;
	}
	
	public int compareTo(Attribute that) {
		if (this.type != that.type) {
			throw new IllegalStateException("Attempt to compare a numeric attribute to a categorical attribute in compareTo method");
		}
		// MISSING > Having a value
		if (this.missing && that.missing) {
			return 0;
		}
		if (this.missing && !that.missing) {
			return 1;
		}
		if (!this.missing && that.missing) {
			return -1;
		}
		if (this.type == Type.Numeric) {
			return Double.compare(this.numericValue, that.numericValue);
		}
		if (this.type == Type.Categorical) {
			return this.categoricalValue.compareTo(that.categoricalValue);
		}
		throw new IllegalStateException("Unsupported attribute type in Attribute.compareTo " + this.type.name());
	}
}
