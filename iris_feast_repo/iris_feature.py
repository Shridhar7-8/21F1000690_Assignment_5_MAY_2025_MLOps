from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64
from datetime import timedelta

# Define the Entity for the Iris data point
iris_entity = Entity(
    name="iris_id", 
    description="A unique ID for each Iris data point", 
    value_type=ValueType.INT64
)

# Define the offline data source using the uploaded CSV file
iris_file_source = FileSource(
    path="../data/iris_data_adapted_for_feast.parquet", 
    timestamp_field="event_timestamp",
)

# Define a Feature View for the four numerical features
iris_stats_fv = FeatureView(
    name="iris_stats",
    entities=[iris_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_file_source,
)