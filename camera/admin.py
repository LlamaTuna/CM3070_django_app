from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, Event, Face

class CustomUserAdmin(UserAdmin):
    """
    Custom admin class for CustomUser model.
    Extends the built-in UserAdmin to include the 'role' field.
    """
    fieldsets = UserAdmin.fieldsets + (
        (None, {'fields': ('role',)}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        (None, {'fields': ('role',)}),
    )

admin.site.register(CustomUser, CustomUserAdmin)

class EventAdmin(admin.ModelAdmin):
    """
    Admin class for the Event model.
    Customizes the admin interface for Event by displaying
    timestamp, event_type, and description in the list view.
    """
    list_display = ('timestamp', 'event_type', 'description')
    search_fields = ('event_type', 'description')
    list_filter = ('event_type', 'timestamp')

admin.site.register(Event, EventAdmin)

class FaceAdmin(admin.ModelAdmin):
    """
    Admin class for the Face model.
    Customizes the admin interface for Face by displaying
    name, timestamp, and tagged status in the list view.
    """
    list_display = ('name', 'timestamp', 'tagged')
    search_fields = ('name',)
    list_filter = ('tagged', 'timestamp')

admin.site.register(Face, FaceAdmin)
