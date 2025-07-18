from django.contrib import admin
from .models import XGBoostModels

@admin.register(XGBoostModels)
class XGBoostModelsAdmin(admin.ModelAdmin):
    """
    Admin interface for the XGBoostModels model.
    Customize the fields and list display if needed.
    """
    list_display = ('id', 'name', 'instrument_key', 'interval', 'created_at')
    search_fields = ('name', 'instrument_key', 'interval')
    list_filter = ('name','instrument_key','interval')
