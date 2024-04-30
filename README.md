# SCVQA-web

Screen Content Video Quality Assessment Website

# Setup for Anaconda env

```bash
conda remove -n SCVQA-webenv --all

conda create -n SCVQA-webenv python=3.9.18

conda activate SCVQA-webenv

conda config --env --add channels conda-forge

conda install -c conda-forge ffmpeg

conda install django

pip install -r requirements.txt

django-admin startproject website .

python manage.py startapp scvqa

python manage.py runserver

# conda deactivate
```
