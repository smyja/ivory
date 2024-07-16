import json
from django.shortcuts import render
from rest_framework import generics, permissions, status,Response
from rest_framework.decorators import (api_view, authentication_classes,
                                     permission_classes)
from rest_framework.permissions import AllowAny,IsAuthenticated
from django.http import JsonResponse



from rest_framework.decorators import api_view
from .models import DataSource, DataPoint
from .serializers import DataSourceSerializer, DataPointSerializer
from datasets import load_dataset
import pandas as pd
import os

PROJECT_DIR = os.path.expanduser('~/my_project')

def index(request):
    return render(request, "index.html")

@api_view(['POST'])
def add_huggingface_dataset(request):
    dataset_name = request.data.get('dataset_name')
    if not dataset_name:
        return Response({"error": "Dataset name is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        dataset = load_dataset(dataset_name, cache_dir=PROJECT_DIR)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    data_source = DataSource.objects.create(name=dataset_name, description=f"Dataset from Hugging Face: {dataset_name}")
    for split in dataset.keys():
        for data in dataset[split]:
            DataPoint.objects.create(source=data_source, data=data)
    
    return Response({"message": "Dataset added successfully"}, status=status.HTTP_201_CREATED)

@api_view(['POST'])
def add_csv_dataset(request):
    file_url = request.data.get('file_url')
    if not file_url:
        return Response({"error": "File URL is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        df = pd.read_csv(file_url)
        df.to_csv(os.path.join(PROJECT_DIR, os.path.basename(file_url)), index=False)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    data_source = DataSource.objects.create(name=file_url.split('/')[-1], description="CSV Dataset")
    for _, row in df.iterrows():
        DataPoint.objects.create(source=data_source, data=row.to_dict())

    return Response({"message": "CSV dataset added successfully"}, status=status.HTTP_201_CREATED)

@api_view(['POST'])
def add_json_dataset(request):
    file_url = request.data.get('file_url')
    if not file_url:
        return Response({"error": "File URL is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        df = pd.read_json(file_url, lines=True)
        df.to_json(os.path.join(PROJECT_DIR, os.path.basename(file_url)), orient='records', lines=True)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    data_source = DataSource.objects.create(name=file_url.split('/')[-1], description="JSON Dataset")
    for _, row in df.iterrows():
        DataPoint.objects.create(source=data_source, data=row.to_dict())

    return Response({"message": "JSON dataset added successfully"}, status=status.HTTP_201_CREATED)

@api_view(['POST'])
def add_parquet_dataset(request):
    file_url = request.data.get('file_url')
    if not file_url:
        return Response({"error": "File URL is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        df = pd.read_parquet(file_url)
        df.to_parquet(os.path.join(PROJECT_DIR, os.path.basename(file_url)))
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    data_source = DataSource.objects.create(name=file_url.split('/')[-1], description="Parquet Dataset")
    for _, row in df.iterrows():
        DataPoint.objects.create(source=data_source, data=row.to_dict())

    return Response({"message": "Parquet dataset added successfully"}, status=status.HTTP_201_CREATED)
