# Generated by Django 4.2.4 on 2023-11-14 11:54

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DFModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_start', models.CharField(max_length=255)),
                ('time_seconds', models.FloatField()),
                ('idx', models.IntegerField()),
                ('right_ear', models.IntegerField()),
                ('frequency', models.FloatField()),
                ('pitch', models.FloatField()),
                ('dB_estimated', models.FloatField()),
                ('dB_start', models.FloatField()),
                ('dB_per_sec', models.FloatField()),
                ('time_detected', models.FloatField()),
                ('error_code', models.IntegerField()),
                ('name', models.CharField(max_length=255)),
                ('user_id', models.CharField(max_length=255)),
                ('dob', models.IntegerField()),
                ('device', models.CharField(max_length=255)),
            ],
        ),
    ]