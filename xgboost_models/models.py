from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class XGBoostModels(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(null=True, blank=True)
    instrument_key = models.CharField(max_length=50)

    INTERVAL_CHOICES = [
        ('1M', '1 Minute'),
        ('15M', '15 Minutes'),
        ('1H', '1 Hour'),
        ('3H', '3 Hours'),
        ('1D', '1 Day')
    ]
    interval = models.CharField(max_length=10, choices=INTERVAL_CHOICES)

    from_date = models.DateField()
    to_date = models.DateField()

    prediction_step = models.PositiveIntegerField()
    train_split = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    eta = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    subsample = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    colsample_bytree = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    seed = models.IntegerField()

    max_depth = models.PositiveIntegerField()
    num_boost_round = models.PositiveIntegerField()
    early_stopping_rounds = models.PositiveIntegerField()
    verbose_eval = models.BooleanField(default=False)
    lagging_count = models.PositiveIntegerField()

    model_file = models.CharField(max_length=255, null=True, blank=True)
    model_structure_file = models.CharField(max_length=255, null=True, blank=True)

    selected_features = models.JSONField(default=dict)  # To store hyperparameters and full_features

    training_mse = models.FloatField(null=True, blank=True)
    training_mae = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
