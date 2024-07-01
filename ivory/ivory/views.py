import json
from django.shortcuts import render
from rest_framework import generics, permissions, status
from rest_framework.decorators import (api_view, authentication_classes,
                                     permission_classes)
from rest_framework.permissions import AllowAny,IsAuthenticated
from django.http import JsonResponse

def index(request):
    return render(request, "index.html")

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def load_huggingface(request):
    try:
        data = json.loads(request.body)
       
        auth_code = data.get('code')

        if not auth_code:
            return JsonResponse({"error": "Authorization code not provided"}, status=400)


        if response.status_code == 200:
            return JsonResponse({"message": "Huggingface connection initiated"},status=200)
        else:
            return JsonResponse({"error": "Failed to initiate Huggingface connection"}, status=500)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)