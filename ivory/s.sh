#!/bin/bash

# Create virtualenv 
project_dir=$(pwd)
virtualenv venv
. ./venv/bin/activate

# Define colors
red=$(tput setaf 1)
green=$(tput setaf 2) 
blue=$(tput setaf 4)
magenta=$(tput setaf 5)
cyan=$(tput setaf 6)
reset=$(tput sgr0)

# Download gitignore and install Django
curl https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore > .gitignore
pip install django
pip install black
# Create Django project
echo "${green}Installing Django${reset}"
echo "${blue}Enter a Project name${reset}"
read project_name
echo "${cyan}Creating Django project${reset}"
django-admin startproject $project_name .
echo "Project created"

# Create Django app
echo "${green}Creating a new Django app, Enter an app name${reset}"
read app_name
python manage.py startapp $app_name
echo "${magenta}Creating a new Django app${reset}"

# Add app to settings
echo "${green}Adding app to settings.py${reset}"
cd $project_name  # Change working directory to the project directory
sed -i "" "/django.contrib.staticfiles/a\\
    '$app_name'," settings.py
echo "${magenta}App added to settings.py${reset}"
cd ..

# Create template
echo "Creating a new template....."
mkdir $app_name/templates
echo "Hello world" > $app_name/templates/index.html
echo "${blue}Template created${reset}"

# Create view 
echo "Creating a new view in the app to display the template"
touch $app_name/views.py
echo "from django.shortcuts import render" >> $app_name/views.py
echo "def index(request):" >> $app_name/views.py
echo " return render(request, 'index.html')" >> $app_name/views.py
echo "${green}View created${reset}"

# Configure app URLs
echo "${green}Adding app to urls.py......${reset}"
echo "from django.urls import path" > $app_name/urls.py 
echo "from $app_name import views" >> $app_name/urls.py
echo "urlpatterns = [" >> $app_name/urls.py
echo "    path('', views.index, name='index')" >> $app_name/urls.py
echo "]" >> $app_name/urls.py
echo "${blue}App added to urls.py${reset}"

# Add the app URLs to the project's urls.py
# Create a new urls.py file with the correct content
echo "${green}Adding app to project urls.py......${reset}"
echo "from django.contrib import admin" >> $project_name/urls.py
echo "from django.urls import path, include" >> $project_name/urls.py
echo -e '\nurlpatterns = [' >> $project_name/urls.py
echo "    path('admin/', admin.site.urls, name='admin')," >> $project_name/urls.py
echo "    path('', include('$app_name.urls'))," >> $project_name/urls.py
echo "]" >> $project_name/urls.py
echo "" >> $project_name/urls.py  # Add an empty line at the end

# Format code using black
echo "${cyan}Formatting code using black${reset}"
black .
echo "${green}Code formatting completed${reset}"
# Finish
python manage.py migrate
echo "${red}F${green}I${blue}N${magenta}I${red}S${green}H${blue}E${magenta}D ${blue}S${green}E${blue}T${magenta}U${blue}P🎉🎉${reset}"
tput init