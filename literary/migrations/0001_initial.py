# Generated by Django 4.1.4 on 2022-12-26 20:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Literarysource',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.TextField(max_length=500, verbose_name='title')),
                ('author', models.TextField(max_length=500, verbose_name='author')),
                ('url_q', models.URLField(max_length=500, verbose_name='url')),
                ('pdf_file', models.FileField(upload_to='', verbose_name='file')),
            ],
        ),
        migrations.CreateModel(
            name='Textesource',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('essay', models.FileField(upload_to='', verbose_name='file')),
                ('essay_post', models.SlugField(max_length=500, verbose_name='text_post')),
                ('project', models.CharField(max_length=100, verbose_name='project')),
            ],
        ),
    ]
