# Generated by Django 4.1.4 on 2022-12-27 00:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('literary', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='textesource',
            name='essay_post',
            field=models.TextField(max_length=500, verbose_name='text_note'),
        ),
        migrations.AlterField(
            model_name='textesource',
            name='project',
            field=models.TextField(max_length=100, verbose_name='project'),
        ),
    ]