from pyspark.sql import functions as F
import matplotlib.pyplot as plt


def print_missing_value_counts(df):
    null_counts = df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])
    null_counts.show(vertical=True)


def mean_residuals(predictions, actuals):
    df = predictions.join(actuals)
    df = df.withColumn('Residual', F.col('prediction') - F.col('LOS'))
    # Calculate mean of residuals
    mean_residual = df.select(F.avg('Residual')).first()[0]

    return abs(round(mean_residual, 5))


def print_metrics(evaluator, predictions):
    mean_residual = mean_residuals(predictions.select("prediction"), predictions.select("LOS"))
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})

    print(
        f"Mean of the difference between predicted and actual 'LOS': {mean_residual}\n"
        f"R2 score (proportion of variance predictable): {r2:.2f}\n"
        f"Root Mean Squared Error (average magnitude of errors): {rmse:.2f}\n"
        f"Mean Squared Error (average of the squares of errors): {mse:.2f}\n"
    )


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
        return list(unique_values)
    return value_list


def replace_empty_list(value_list):
    if not value_list:
        return ["UNKNOWN"]
    return value_list


def plot_graph(df, group_by_column, aggregate_column, agg_func, plot_title, x_label, y_label):
    aggregated_data = df.groupBy(group_by_column).agg(agg_func(aggregate_column).alias('agg_result')).collect()

    categories = [row[group_by_column] for row in aggregated_data]
    values = [row['agg_result'] for row in aggregated_data]

    colors = [plt.cm.tab10(i / len(categories)) for i in range(len(categories))]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.show()
