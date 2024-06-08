from pyspark.sql import functions as F
import matplotlib.pyplot as plt


def print_missing_value_counts(df):
    # Calculate null counts for each column
    null_counts = df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])

    # To display in a more readable vertical format
    null_counts.show(vertical=True)


def classify_columns(df):
    categorical_cols = []
    numerical_cols = []

    for field in df.schema.fields:
        column_name = field.name
        first_non_null = df.filter(df[column_name].isNotNull()).select(column_name).first()

        if first_non_null is not None:
            value = first_non_null[0]
            if isinstance(value, (str, int, float)):  # Ensure the value is a basic datasets type suitable for conversion
                try:
                    # First, try to convert the value to int
                    _ = int(value)
                    numerical_cols.append(column_name)  # Success means it's numerical
                except ValueError:
                    try:
                        # If int fails, try to convert to float
                        _ = float(value)
                        numerical_cols.append(column_name)  # Success means it's numerical
                    except ValueError:
                        # If both conversions fail, classify as categorical
                        categorical_cols.append(column_name)
            else:
                # Handle non-string, non-numeric types (like lists, dicts, etc.)
                categorical_cols.append(column_name)
        else:
            # If all values are null, consider it categorical for safety
            categorical_cols.append(column_name)

    return numerical_cols, categorical_cols



# Define the UDF
def handle_list(value):
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
    return value

def transform_list(value_list):
    if value_list is None:
        return None
    unique_values = set(value_list)
    if len(unique_values) == 1:
        # Return the single element wrapped in an array
        return list(unique_values)
    return value_list  # Return the original list if values are not the same


# Define the UDF to replace empty lists with 'UNKNOWN'
def replace_empty_list(value_list):
    if not value_list:  # Checks if the list is empty
        return ["UNKNOWN"]
    return value_list


def plot_graph(df, group_by_column, aggregate_column, agg_func, plot_title, x_label, y_label):
    # Aggregate the datasets
    aggregated_data = df.groupBy(group_by_column).agg(agg_func(aggregate_column).alias('agg_result')).collect()

    # Extracting results into lists for plotting
    categories = [row[group_by_column] for row in aggregated_data]
    values = [row['agg_result'] for row in aggregated_data]

    # Choose random colors for the bars
    colors = [plt.cm.tab10(i/len(categories)) for i in range(len(categories))]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)  # Adjust rotation based on category name length
    plt.show()


"""
def classify_disease_risk(df, disease_col, los_col):
    # Alias the DataFrame to avoid ambiguity in join operations
    df_aliased = df.alias("df_main")

    # Explode the disease codes to analyze their individual impact on LOS
    df_exploded = df_aliased.selectExpr("explode({}) as Disease".format(disease_col), los_col)

    # Calculate average LOS for each disease
    avg_los_by_disease = df_exploded.groupBy("Disease").agg(avg(los_col).alias("Avg_LOS"))

    # Get max and min LOS for scaling
    max_los = avg_los_by_disease.agg({"Avg_LOS": "max"}).collect()[0][0]
    min_los = avg_los_by_disease.agg({"Avg_LOS": "min"}).collect()[0][0]

    # Define a UDF to scale LOS to a risk score from 1 to 10
    def scale_los_to_risk(los):
        return 1 + int((los - min_los) / (max_los - min_los) * 9) if max_los > min_los else 1

    scale_los_to_risk_udf = udf(scale_los_to_risk, IntegerType())

    # Apply the UDF to calculate risk scores
    risk_scores = avg_los_by_disease.withColumn("Risk_Score", scale_los_to_risk_udf(col("Avg_LOS")))

    # Join the risk scores back with the exploded DataFrame to assign scores to each disease
    df_with_risk = df_exploded.join(risk_scores, "Disease")

    # Group by original DataFrame IDs and calculate average risk score
    df_risk_value = df_with_risk.groupBy(df_aliased.columns).agg(avg("Risk_Score").alias("RISK_VALUE"))

    # Drop the disease code column and return the result
    result_df = df_risk_value.drop(disease_col)

    return result_df
"""