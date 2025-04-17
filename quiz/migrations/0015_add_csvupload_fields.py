from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quiz', '0014_auto_20250417_1307'),  # Updated to match the latest migration
    ]

    operations = [
        migrations.AddField(
            model_name='csvupload',
            name='cleaned_data',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='csvupload',
            name='model_trained',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='csvupload',
            name='model_accuracy',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='csvupload',
            name='insights',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='csvupload',
            name='stored_filename',
            field=models.CharField(default='', max_length=255),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='csvupload',
            name='file_size',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ] 