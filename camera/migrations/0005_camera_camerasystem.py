# Generated by Django 4.2 on 2024-07-28 22:52

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('camera', '0004_alter_emailsettings_smtp_password_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Camera',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('url', models.URLField()),
            ],
        ),
        migrations.CreateModel(
            name='CameraSystem',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('camera1', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='camera1', to='camera.camera')),
                ('camera2', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='camera2', to='camera.camera')),
                ('camera3', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='camera3', to='camera.camera')),
                ('camera4', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='camera4', to='camera.camera')),
            ],
        ),
    ]
