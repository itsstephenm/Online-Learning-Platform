from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('quiz', '0015_add_csvupload_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='csvupload',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='csv_uploads', to=settings.AUTH_USER_MODEL),
        ),
    ] 