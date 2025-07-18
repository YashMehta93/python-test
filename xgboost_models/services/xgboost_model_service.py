from ..repository.xgboost_model_repo import XGBoostModelRepository
from ..ml.xgboost_trainer import train_model


class XGBoostModelService:
    @staticmethod
    def get_all_models():
        return XGBoostModelRepository.get_all_models()

    @staticmethod
    def get_model_by_id(pk):
        return XGBoostModelRepository.get_model_by_id(pk)

    @staticmethod
    def save_model_info(model_info):
        return XGBoostModelRepository.save_model_info(model_info)

    @staticmethod
    def update_model_info(pk, updated_info):
        return XGBoostModelRepository.update_model_info(pk, updated_info)

    @staticmethod
    def delete_model_by_id(pk):
        return XGBoostModelRepository.delete_model_by_id(pk)

    
def update_model_training_results(model_id, model_path, model_structure_path, training_mse, training_mae, selected_features):
    model = XGBoostModelService.get_model_by_id(model_id)
    model.model_file = model_path
    model.model_structure_file = model_structure_path
    model.training_mse = training_mse
    model.training_mae = training_mae
    model.selected_features = selected_features
    model.save()


def mark_model_as_failed(model_id, error_message):
    model = XGBoostModelService.get_model_by_id(model_id)
    model.description = f"Training failed: {error_message}"
    model.save()
