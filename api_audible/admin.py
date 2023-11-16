from django.contrib import admin
from .models import DFModel
from import_export.admin import ImportExportModelAdmin


class DFModelAdmin(ImportExportModelAdmin):
    list_display = ('name', 'user_id', 'dob', 'device', 'session_start', 'time_seconds', 'idx', 'right_ear', 'frequency', 'pitch', 'dB_estimated', 'dB_start', 'dB_per_sec', 'time_detected', 'error_code')
    search_fields = ('name', 'user_id', 'session_start','device')  # Optional: Add fields for searching

# Register your models here.
admin.site.register(DFModel,DFModelAdmin)

