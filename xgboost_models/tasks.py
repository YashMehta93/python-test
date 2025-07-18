from celery import shared_task
from .services.xgboost_model_service import update_model_training_results, mark_model_as_failed
from .ml.xgboost_trainer import train_model 


@shared_task
def async_train_model(
    model_id,
    instrument_key,
    interval,
    from_date,
    to_date,
    prediction_step,
    train_split,
    max_depth,
    eta,
    subsample,
    colsample_bytree,
    seed,
    name,
    lagging_count,
    num_boost_round,
    early_stopping_rounds,
    verbose_eval,
    selected_features
):
    try:
        # Run the training process
        trained_model = train_model(
            instrument_key=instrument_key,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            prediction_step=prediction_step,
            train_split=train_split,
            max_depth=max_depth,
            eta=eta,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            seed=seed,
            name=name,
            lagging_count=lagging_count,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            selected_features=selected_features
        )

        # Update the model record in the database
        update_model_training_results(
            model_id=model_id,
            model_file=trained_model.model_file,
            model_structure_path=trained_model.model_structure_file,
            training_mse=trained_model.training_mse,
            training_mae=trained_model.training_mae,
            selected_features=trained_model.selected_features
        )
    except Exception as e:
        # Log the error and mark the model as failed
        mark_model_as_failed(model_id, error_message=str(e))
