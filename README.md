# djangoproject_macOS
Part of Literarywebapp with Deep Learning methods Important content:

    "texteapp"-directory is a core-app for manager and maitanance, content: 1.1. settings.py; 1.2. urls.py

    "literary"- directory is a program for textesearch, content: 2.1. "templates"-directory with index.html; 2.2. "fixtures" directory with mypapers.json; 2.3. models.py with database class; 2.4. views.py mixins and generic class based views;

    "Mytext"- directory is a program for projectmanagement and textesearch (now not completely, but prototype): 3.1."templates"-directory with index.html; 3.2. "fixtures" directory with myproject.json and mytext.json; 3.3. models.py with database class; 2.4. views.py mixins and generic class based views;

    db.sqlite3 is database

Requierements:

Python 3.9.12 Django 4.1.1 Pip 22.3.1 Supplementary Library, install with pip: django-easticrearch-dsl; NLTK; pandas; spaCy; TextBlob; numpy; docx; bs4; gensim; djangorestframework; django-filter; scikit-learn; unidecode; tweepy; seaborn; sklearn_crfsuite; drf-json-api; djangorestframework-jsonapi; PyPDF2(deprecated at 24.12.2022) replace with pdf; pdfminer.six; matplotlib; HanTa; Djangoproject_state_of_the_art: TensorFlow; biobert_bern (https://pypi.org/project/biobert-bern/); transformers;python-heatclient; seaborn; matplotlib;mpi;venv¿ User manual (MAC OS):

    git clone -link
    %cd djangoproject
    %python –m venv venv (=create the venv)
    % source venv/bin/activate  (=activate the venv)
    pip install --upgrade pip
    pip install django
    Open the texteapp/settings.py; Enter the ALLOWED_HOSTS = ['127.0.0.1','localhost',], save
    %python manage.py createsuperuser
    { Enter username (Olgjanius) and password (m57BOG+P)
    %python manage.py runserver (=start server)
    Start webbrowser, localhost:8000/admin/
    Start search for paper under: localhost:8000/search/

User manual (Windows):

    git clone -link

        cd djangoproject

        py --version (=control of python)

        py -m venv venv (=create of virtual enviroment)

        venv\Scripts\activate.bat (=activate of virtual enviroment)

    py -m pip install --upgrade pip
    py -m pip install Django
    Open the texteapp/settings.py; Enter the ALLOWED_HOSTS = ['127.0.0.1','localhost',], save

        python manage.py createsuperuser

    { Enter username () and password ()

        python manage.py runserver (=start server)

    Start webbrowser, localhost:8000/admin/
    Start search for paper under: localhost:8000/search/ Schema:

