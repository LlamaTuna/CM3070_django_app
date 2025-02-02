# Generated by Django 4.2 on 2024-07-30 03:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('camera', '0006_remove_camerasystem_camera1_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Camera',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('url', models.CharField(max_length=200)),
                ('type', models.CharField(choices=[('ip', 'IP Camera'), ('usb', 'USB Camera')], max_length=3)),
            ],
        ),
    ]
