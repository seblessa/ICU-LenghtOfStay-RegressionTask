from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import seaborn as sns


def print_missing_value_counts(df):
    # Calculate null counts for each column
    null_counts = df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])

    # To display in a more readable vertical format
    null_counts.show(vertical=True)


def mean_residuals(predictions, actuals):
    # Ensure predictions and actuals are columns of DataFrames
    df = predictions.join(actuals)
    df = df.withColumn('Residual', F.col('prediction') - F.col('LOS'))  # Ensure correct column names are used

    # Calculate mean of residuals
    mean_residual = df.select(F.avg('Residual')).first()[0]

    return abs(round(mean_residual, 2))


def print_metrics(evaluator, predictions):
    # Mean of differences between predicted and actual values
    mean_residual = mean_residuals(predictions.select("prediction"), predictions.select("LOS"))
    print(f"Mean of the difference between predicted 'LOS' and actual 'LOS': {mean_residual}.")

    # R2 Score
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    print(f"R2 score, indicating the proportion of variance in 'LOS' that is predictable from the features: {r2:.2f}")

    # Root Mean Squared Error (RMSE)
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    print(f"Root Mean Squared Error, showing the average magnitude of the prediction errors in 'LOS': {rmse:.2f}")

    # Mean Squared Error (MSE)
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
    print(f"Mean Squared Error, representing the average of the squares of the prediction errors in 'LOS': {mse:.2f}")


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
    colors = [plt.cm.tab10(i / len(categories)) for i in range(len(categories))]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.show()
