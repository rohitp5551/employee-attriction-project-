from databricks.sdk.runtime import spark

import pandas as pd
def load_data():
    storage_account_name="rohitmlemployee"
    storage_account_key = "ADqXu5fJgAbLtkSB5ktmT4HdWLap6/KdTNSnZA+AblPk9+qhO09x0QbyMNW9ljpvsThRFleCkIq8+AStQS+bSQ=="
    spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",storage_account_key)
    df=spark.read.csv(f"wasbs://bronze@rohitmlemployee.blob.core.windows.net/WA_Fn-UseC_-HR-Employee-Attrition.csv",header=True,inferSchema=True)
    df=df.toPandas()
    return df