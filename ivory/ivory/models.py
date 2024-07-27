from django.db import models

class DataSource(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class DataPoint(models.Model):
    source = models.ForeignKey(DataSource, on_delete=models.CASCADE, related_name='data_points')
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"DataPoint {self.id} from {self.source.name}"

class Cluster(models.Model):
    name = models.CharField(max_length=100)
    data_points = models.ManyToManyField(DataPoint, related_name='clusters')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class ClusteringTask(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    status = models.CharField(max_length=50, choices=[('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed')])
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.name
