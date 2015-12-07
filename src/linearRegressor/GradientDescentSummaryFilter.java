package linearRegressor;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeSet;

import com.google.common.collect.ImmutableMap;

import utilities.StopWatch;


public class GradientDescentSummaryFilter {	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((filter == null) ? 0 : filter.hashCode());
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		GradientDescentSummaryFilter other = (GradientDescentSummaryFilter) obj;
		if (filter == null && other.filter == null) {
			return true;
		} else if ((filter == null && other.filter != null) || (filter != null && other.filter == null)) {
			return false;
		} else if (!filter.keySet().containsAll(other.filter.keySet()) || !other.filter.keySet().containsAll(filter.keySet())) {
				return false;
		} else {
			for (FilterableProperty thisKey : filter.keySet()) {
				if (!doPropertiesMatch(thisKey, filter.get(thisKey), other.filter.get(thisKey))) {
					return false;
				}
			}
		}
		return true;
	}
	public enum FilterableProperty {UpdateRule}
	
	public Map<FilterableProperty, Object> filter;
	
	public GradientDescentSummaryFilter(Map<FilterableProperty, Object> filter) {
		this.filter = filter;
	}
	/**
	 * Use to concatenate predefined filters;
	 * @param filters
	 */
	@SafeVarargs
	public GradientDescentSummaryFilter(Map<FilterableProperty, Object>... filters) {
		this.filter = new HashMap<FilterableProperty, Object>();
		for (Map<FilterableProperty, Object> map : filters) {
			this.filter.putAll(map);
		}
	}
	
	public GradientDescentSummaryFilter(GradientDescentSummaryFilter... filters) {
		this.filter = new HashMap<FilterableProperty, Object>();
		for (GradientDescentSummaryFilter filter : filters) {
			this.filter.putAll(filter.filter);
		}
	}
	
	public GradientDescentSummaryFilter(FilterableProperty[] properties, Object[] values) {
		if (properties.length != values.length) {
			throw new IllegalArgumentException();
		}
		
		this.filter = new HashMap<FilterableProperty, Object>();
		for (int i = 0; i < properties.length; i++) {
			this.filter.put(properties[i], values[i]);
		}
	}
	
	public String getLongFilterDescription() {
		StringBuffer description = new StringBuffer();
		for (Map.Entry<FilterableProperty, Object> filterEntry : filter.entrySet()) {
			if (description.length() > 0) {
				description.append(" and ");
			}
			description.append(filterEntry.getKey().name() + "=" + filterEntry.getValue().toString());
		}
		return description.toString();
	}
	
	public String getMinimalFilterDescription() {
		StringBuffer description = new StringBuffer();
		for (Map.Entry<FilterableProperty, Object> filterEntry : filter.entrySet()) {
			if (description.length() > 0) {
				description.append("and");
			}
			description.append(getMinimizedPropertyName(filterEntry.getKey()) + "=" + filterEntry.getValue().toString());
		}
		return description.toString();
	}
	
	public TreeSet<GradientDescentSummary> filterRecordsOnParameterValue(TreeSet<GradientDescentSummary> allRecords) {
		if (filter == null) {
			return allRecords;
		}
		HashSet<GradientDescentSummary> filteredRecords = new HashSet<>(allRecords);
		
		for (Map.Entry<FilterableProperty, Object> filterEntry : filter.entrySet()) {
			HashSet<GradientDescentSummary> recordsToRemove = new HashSet<>();
			for (GradientDescentSummary record : filteredRecords) {
				if (!doesRecordMatchFilter(record, filterEntry)) {
					recordsToRemove.add(record);
				}
			}
			filteredRecords.removeAll(recordsToRemove);
		}
		
		return new TreeSet<GradientDescentSummary>(filteredRecords);
	}
	
	public String getSubDirectory() {
		StringBuffer buffer = new StringBuffer();
		TreeSet<FilterableProperty> sortedKeys = new TreeSet<>(filter.keySet());
		for (FilterableProperty property : sortedKeys) {
			Object value = filter.get(property);
			double doubleValue = Double.NaN;
			try {
				doubleValue = (double)value;
			} catch (ClassCastException e) {}
			String stringValue = value.toString();
			if (!Double.isNaN(doubleValue)) {
				stringValue = String.format("%f", doubleValue);
			}
			buffer.append(getMinimizedPropertyName(property) + "-" + stringValue + "/");
		}
		return buffer.toString();
	}
	
	private static String getMinimizedPropertyName(FilterableProperty property) {
		switch(property) {
			case UpdateRule:
				return "UR";
			default:
				throw new IllegalArgumentException();
		
		}
	}
	
	private static boolean doesRecordMatchFilter(GradientDescentSummary record, Map.Entry<FilterableProperty, Object> filterEntry) {
		try {
			String cast = (String)filterEntry.getValue();
			if (cast.equals("ALL")) {
				return true;
			} else {
				throw new IllegalArgumentException("Only String filter value supported is \"All\"");
			}
		} catch (ClassCastException e) {
			// No Problem keep checking.
		}
		switch (filterEntry.getKey()) {
			case UpdateRule:
				return record.parameters.updateRule == (UpdateRule)filterEntry.getValue();
		}
		System.err.println(StopWatch.getDateTimeStamp() + "ERROR: Shouldn't reach here in RunDataSummaryRecord.doesRecordMatchFilter");
		return false;
	}
	
	private static boolean doPropertiesMatch(FilterableProperty property, Object value1, Object value2) {
		String cast1 = null, cast2 = null;
		try {
			cast1 = (String)value1;
		} catch (ClassCastException e) {
			cast1 = null;
		}
		try {
			cast2 = (String)value2;
		} catch (ClassCastException e) {
			cast2 = null;
		}
		if (cast1 != null || cast2 != null) {
			return cast1.equals(cast2);
		}
	
		switch (property) {
			case UpdateRule:
				return (UpdateRule)value1 == (UpdateRule)value2;
		}
		System.err.println(StopWatch.getDateTimeStamp() + "ERROR: Shouldn't reach here in RunDataSummaryRecord.doesRecordMatchFilter");
		return false;
	}
	
	// Predefined Bag Fraction Filters
	public static GradientDescentSummaryFilter updateRuleEqualsOriginal = 
			new GradientDescentSummaryFilter(new ImmutableMap.Builder<FilterableProperty, Object>()
					.put(FilterableProperty.UpdateRule, UpdateRule.Original).build());
	public static GradientDescentSummaryFilter updateRuleEqualsAdaptedLR = 
			new GradientDescentSummaryFilter(new ImmutableMap.Builder<FilterableProperty, Object>()
					.put(FilterableProperty.UpdateRule, UpdateRule.AdaptedLR).build());
	public static GradientDescentSummaryFilter updateRuleEqualsGradientMag = 
			new GradientDescentSummaryFilter(new ImmutableMap.Builder<FilterableProperty, Object>()
					.put(FilterableProperty.UpdateRule, UpdateRule.GradientMag).build());
}
