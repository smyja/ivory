from rest_framework import serializers
from .models import DataSource, DataPoint, ClusteringTask, Cluster

class DataSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSource
        fields = '__all__'

class DataPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataPoint
        fields = '__all__'

class ClusteringTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClusteringTask
        fields = '__all__'

class ClusterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cluster
        fields = '__all__'
