from django.db import models

class DFModel(models.Model):
    session_start = models.CharField(max_length=255)
    time_seconds = models.FloatField()
    idx = models.IntegerField()
    right_ear = models.IntegerField()
    frequency = models.FloatField()
    pitch = models.FloatField()
    dB_estimated = models.FloatField()
    dB_start = models.FloatField()
    dB_per_sec = models.FloatField()
    time_detected = models.FloatField()
    error_code = models.IntegerField()
    name = models.CharField(max_length=255)
    user_id = models.CharField(max_length=255)
    dob = models.IntegerField()
    device = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.name} - {self.session_start}"

